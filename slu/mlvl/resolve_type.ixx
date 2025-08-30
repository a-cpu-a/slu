module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <compare>
#include <variant>
#include <utility>
#include <stdexcept>

#include <slu/ext/CppMatch.hpp>
export module slu.mlvl.resolve_type;
import slu.num;
import slu.ast.mp_data;
import slu.ast.state_decls;
import slu.ast.state;
import slu.ast.type;
import slu.lang.mpc;

namespace slu::mlvl
{
	export parse::ResolvedType resolveTypeExpr(parse::BasicMpDb mpDb, parse::Expr&& type);
	
	void handleTypeExprField(parse::BasicMpDb mpDb, size_t& nameIdx, parse::FieldV<true>& field,auto& res)
	{
		ezmatch(std::move(field))(
		varcase(parse::FieldType::Expr&&) {
			res.fieldNames.emplace_back("0x" + slu::u64ToStr(nameIdx++));
			res.fields.emplace_back(resolveTypeExpr(mpDb, std::move(var)));
		},
			varcase(parse::FieldType::Name2Expr&&) {
			res.fieldNames.emplace_back(var.idx.asSv(mpDb));
			res.fields.emplace_back(resolveTypeExpr(mpDb, std::move(var.v)));
		},
			varcase(parse::FieldType::Expr2Expr&&) {
			throw std::runtime_error("FieldType::Expr2ExprV type resolution: TODO not implemented: jit the expression");
		},
			varcase(const parse::FieldType::NONE) {
			throw std::runtime_error("FieldType::NONE should not exist in struct/union type expression.");
		}
		);
	}

	parse::ResolvedType resolveStructType(parse::BasicMpDb mpDb, parse::TableV<true>&& itm)
	{
		parse::StructRawType& res = *(new parse::StructRawType());
		res.name = lang::MpItmId::newEmpty();
		res.fieldNames.reserve(itm.size());
		res.fields.reserve(itm.size());
		res.fieldOffsets.reserve(itm.size());
		size_t idx = 1;
		size_t fieldOffset = 0;
		uint8_t maxAlign = 0;
		for (parse::FieldV<true>& field : itm)
		{
			handleTypeExprField(mpDb, idx, field, res);
			//TODO: fix field offsets changing depending on named field order, and tuple field order between them too.
			if (fieldOffset != parse::ResolvedType::INCOMPLETE_MARK)
				continue;

			const parse::ResolvedType& resField = res.fields.back();
			maxAlign = std::max(maxAlign, (uint8_t)resField.alignmentData);

			if (resField.size == 0)
			{
				res.fieldOffsets.push_back(SIZE_MAX - 1);//undefined value.
				continue;
			}
			if (!resField.isComplete())
			{
				fieldOffset = parse::ResolvedType::INCOMPLETE_MARK;
				continue;
			}
			if (fieldOffset == parse::ResolvedType::UNSIZED_MARK)
			{
				res.fieldOffsets.push_back(SIZE_MAX);//Only known at runtime.
				continue;
			}
			fieldOffset = (fieldOffset + 7) & (~0b111);//ceil to byte boundary.

			res.fieldOffsets.push_back(fieldOffset);
			fieldOffset += resField.size;

			if (!resField.isSized())
				fieldOffset = parse::ResolvedType::UNSIZED_MARK;//Following fields will only have offsets known at runtime.
		}

		return parse::ResolvedType{
			.base = parse::RawTypeKind::Struct(&res),
			.size = fieldOffset,
			.alignmentData = maxAlign
		};
	}
	parse::ResolvedType resolveTypeExpr(parse::BasicMpDb mpDb, parse::Expr&& type)
	{
		parse::ResolvedType resTy = ezmatch(std::move(type.data))(
		varcase(parse::ExprType::ParensV<true>&&) {
			return resolveTypeExpr(mpDb,std::move(*var));
		},
		varcase(const parse::ExprType::Infer) {
			return parse::ResolvedType::getInferred();
		},
		varcase(parse::ExprType::String&&){
			return parse::ResolvedType::getConstType(parse::RawTypeKind::String{std::move(var.v)});
		},
		varcase(const parse::ExprType::I64){
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Int64{ var });
		},
		varcase(const parse::ExprType::U64){
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint64{ var });
		},
		varcase(const parse::ExprType::P128) {
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Pos128{ var.lo,var.hi });
		},
		varcase(const parse::ExprType::M128) {
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Neg128{ var.lo,var.hi });
		},
		varcase(const parse::ExprType::F64)->parse::ResolvedType {
			throw std::runtime_error("TODO: resolve F64 type expressions.");
		},
		varcase(const parse::ExprType::Deref)->parse::ResolvedType {
			throw std::runtime_error("TODO: resolve Deref type expressions.");
		},
		varcase(const parse::ExprType::Index&&)->parse::ResolvedType {
			throw std::runtime_error("TODO: resolve Index type expressions.");
		},
		varcase(const parse::ExprType::FieldV<true>&&)->parse::ResolvedType {
			throw std::runtime_error("TODO: resolve Field type expressions.");
		},
		varcase(const parse::ExprType::Call&&)->parse::ResolvedType {
			throw std::runtime_error("TODO: resolve call type expressions.");
		},
		varcase(parse::ExprType::FnType&&)->parse::ResolvedType {
			throw std::runtime_error("TODO: resolve Fn type expressions.");
		},
		varcase(parse::ExprType::Dyn&&)->parse::ResolvedType {
			throw std::runtime_error("TODO: resolve Dyn type expressions.");
		},
		varcase(parse::ExprType::Impl&&)->parse::ResolvedType {
			throw std::runtime_error("TODO: resolve Impl type expressions.");
		},
		varcase(parse::ExprType::Err&&)->parse::ResolvedType {
			//TODO replace with ?~~var equivelant.
			throw std::runtime_error("TODO: resolve Err type expressions.");
		},

		varcase(parse::ExprType::SelfCall&&)->parse::ResolvedType {
			//TODO: resolve basic ops, jit all else.
			auto& expArgs = std::get<parse::ArgsType::ExprList>(var.args);
			if(expArgs.size()!=0)
				throw std::runtime_error("TODO: self call type expressions with >0 arg.");

			parse::ResolvedType rt= resolveTypeExpr(mpDb, std::move(*var.v));

			if (var.method == mpDb.data->getItm({ "std","ops","MarkMut","markMut" }))
			{
				if(rt.hasMut)
					throw std::runtime_error("Invalid type expression: used 'mut' on already marked as mutable type.");
				rt.hasMut = true;
				return rt;
			}
			ast::UnOpType op;
			if(var.method == mpDb.data->getItm({ "std","ops","Ref","ref" }))
				op = ast::UnOpType::REF;
			else if (var.method == mpDb.data->getItm({ "std","ops","Ptr","ptr" }))
				op = ast::UnOpType::PTR;
			else
				throw std::runtime_error("Unimplemented type expression: " + std::string(var.method.asSv(mpDb)) + " (type resolution)");

			const bool zst = rt.size == 0;

			//TODO apply lifetime.

			if (rt.outerSliceDims != 0)
			{
				size_t sz = zst ? 0 : (parse::TYPE_RES_PTR_SIZE + parse::TYPE_RES_SIZE_SIZE * 2 * rt.outerSliceDims);

				return parse::ResolvedType{
					.base = parse::RawTypeKind::RefSlice{new parse::RefSliceRawType{
						.elem=std::move(rt),
						.refType= op
					}},
					.size = sz,
					.alignmentData=parse::alignDataFromSize(sz)
				};
			}
			if (std::holds_alternative<parse::RawTypeKind::RefChain>(rt.base))
			{
				//If already a ref chain, then just add the sigil.
				auto& refChain = std::get<parse::RawTypeKind::RefChain>(rt.base);
				refChain->chain.push_back(parse::RefSigil{ .refType = op });
				return rt;
			}
			size_t sz = zst ? 0 : parse::TYPE_RES_PTR_SIZE;
			return parse::ResolvedType{
				.base = parse::RawTypeKind::RefChain{new parse::RefChainRawType{
					.elem = std::move(rt),
					.chain = { parse::RefSigil{.refType = ast::UnOpType::REF}}
				}},
				.size = sz,
				.alignmentData = parse::alignDataFromSize(sz)
			};
		},
		[&](const parse::ExprType::GlobalV<true> name)->parse::ResolvedType {

			if (name == mpc::STD_STR)
			{
				//TODO: this is wrong, it must be wrapped in a struct!
				return parse::ResolvedType{
					.base = parse::RawTypeKind::Range64{0,UINT8_MAX},
					.size = 8,
					.alignmentData= parse::alignDataFromSize(8),
					.outerSliceDims=1
				};
			}
			if (name == mpc::STD_I32)
			{
				return parse::ResolvedType{
					.base = parse::RawTypeKind::Range64{INT32_MIN,INT32_MAX},
					.size = 32,
					.alignmentData = parse::alignDataFromSize(32),
				};
			}
			if (name == mpc::STD_I8)
			{
				return parse::ResolvedType{
					.base = parse::RawTypeKind::Range64{INT8_MIN,INT8_MAX},
					.size = 8,
					.alignmentData = parse::alignDataFromSize(8),
				};
			}
			if (name == mpc::STD_U8)
			{
				return parse::ResolvedType{
					.base = parse::RawTypeKind::Range64{0,UINT8_MAX},
					.size = 8,
					.alignmentData = parse::alignDataFromSize(8),
				};
			}

			throw std::runtime_error("TODO: resolve complex lim-prefix-expr type expressions.");
		},
		varcase(parse::ExprType::Slice&&)->parse::ResolvedType {
			parse::ResolvedType rt = resolveTypeExpr(mpDb, std::move(*var.v));
			rt.outerSliceDims++;
			return rt;
		},
		varcase(parse::ExprType::Struct&&) {
			return resolveStructType(mpDb,std::move(var.fields));
		},
		varcase(parse::ExprType::TableV<true>&&) {
			return resolveStructType(mpDb, std::move(var));
		},
		varcase(parse::ExprType::Union&&)->parse::ResolvedType {
			parse::UnionRawType& res = *(new parse::UnionRawType());
			size_t maxSize=0;
			uint8_t maxAlign = 0;

			size_t idx = 1;
			for (parse::FieldV<true>& field : var.fields)
			{
				handleTypeExprField(mpDb, idx, field, res);
				if (maxSize == parse::ResolvedType::INCOMPLETE_MARK)
					continue;//If any field is incomplete, then union is incomplete.

				const parse::ResolvedType& resField = res.fields.back();
				maxAlign = std::max(maxAlign, (uint8_t)resField.alignmentData);

				if (!resField.isComplete())
				{
					maxSize = parse::ResolvedType::INCOMPLETE_MARK;
					continue;
				}
				if (maxSize == parse::ResolvedType::UNSIZED_MARK)
					continue;//Field is complete, but still the union will be unsized.

				if (!resField.isSized())
				{
					maxSize = parse::ResolvedType::UNSIZED_MARK;
					continue;
				}
				maxSize = std::max(maxSize,resField.size);
			}
			return parse::ResolvedType{
				.base = parse::RawTypeKind::Union(&res),
				.size = maxSize,
				.alignmentData = maxAlign
			};
		},


		varcase(parse::ExprType::IfCondV<true>&&) ->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve if type expressions.");
		},


#define Slu_INVALID_EXPR_CASE(_MSG,...) varcase(__VA_ARGS__ &&)->parse::ResolvedType { \
		throw std::runtime_error("Invalid slu " _MSG ", index:" #__VA_ARGS__ "."); \
	}

		Slu_INVALID_EXPR_CASE("expression",parse::ExprType::OpenRange),
		Slu_INVALID_EXPR_CASE("expression",parse::ExprType::MpRoot),
		Slu_INVALID_EXPR_CASE("expression",parse::ExprType::Local),
		Slu_INVALID_EXPR_CASE("expression",parse::ExprType::True),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::False),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::Nil),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::VarArgs),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::PatTypePrefix),
		Slu_INVALID_EXPR_CASE("type", parse::ExprType::TraitExpr),
		Slu_INVALID_EXPR_CASE("type", parse::ExprType::Function),
		Slu_INVALID_EXPR_CASE("type", parse::ExprType::Lifetime),
		varcase(parse::ExprType::MultiOpV<true>&&)->parse::ResolvedType {
			throw std::runtime_error("Multi-op type expressions are ment to be desuagared before type resolution.");
		});
#undef Slu_INVALID_EXPR_CASE

		return resTy;
	}
}