/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/parser/BasicGenData.hpp>

namespace slu::mlvl
{
	inline const size_t TYPE_RES_SIZE_SIZE = 64;//TODO: unhardcode this, allow 8 bits too.
	inline const size_t TYPE_RES_PTR_SIZE = 64;//TODO: unhardcode this, allow 8 bits too.

	parse::ResolvedType resolveTypeExpr(parse::BasicMpDb mpDb, parse::ExprV<true>&& type)
	{
		//TODO: change mlir conv code to use global itm's.
		parse::ResolvedType resTy = ezmatch(std::move(type.data))(
		varcase(const parse::ExprType::Inferr) {
			return parse::ResolvedType::getInferred();
		},
		varcase(const parse::ExprType::OpenRange)->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve open range type expressions.");
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
		varcase(const parse::ExprType::I128) {
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Int128{ var.lo,var.hi });
		},
		varcase(const parse::ExprType::U128) {
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
		},
		varcase(const parse::ExprType::F64)->parse::ResolvedType {
			throw std::runtime_error("TODO: resolve F64 type expressions.");
		},
		varcase(parse::ExprType::FnType&&)->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve Fn type expressions.");
		},
		varcase(parse::ExprType::Dyn&&)->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve Dyn type expressions.");
		},
		varcase(parse::ExprType::Impl&&)->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve Impl type expressions.");
		},
		varcase(parse::ExprType::Err&&)->parse::ResolvedType {
			//TODO replace with ?~~var equivelant.
			throw std::runtime_error("TODO: resolve Err type expressions.");
		},

		varcase(parse::ExprType::FuncCallV<true>&&)->parse::ResolvedType {
			//TODO: resolve basic ops, jit all else.
			if(var.argChain.size() != 1)
				throw std::runtime_error("TODO: resolve complex FuncCall type expressions.");

			auto& func = std::get<parse::LimPrefixExprType::VARv<true>>(*var.val).v;
			if (!func.sub.empty())
				throw std::runtime_error("Unimplemented type expression (has subvar's) (type resolution)");
			auto name = std::get<parse::BaseVarType::NAMEv<true>>(func.base).v;

			auto& expArgs = std::get<parse::ArgsType::ExprListV<true>>(var.argChain[0].args);
			auto& firstArgExpr = expArgs.front();

			parse::ResolvedType rt= resolveTypeExpr(mpDb, std::move(firstArgExpr));

			if (name == mpDb.data->getItm({ "std","ops","MarkMut","markMut" }))
			{
				if(rt.hasMut)
					throw std::runtime_error("Invalid type expression: used 'mut' on already marked as mutable type.");
				rt.hasMut = true;
				return rt;
			}
			if (name != mpDb.data->getItm({ "std","ops","Ref","ref" }))
				throw std::runtime_error("Unimplemented type expression: " + std::string(name.asSv(mpDb)) + " (type resolution)");

			//TODO apply lifetime.

			if (rt.outerSliceDims != 0)
			{
				for (size_t i = 0; i < rt.outerSliceDims; i++)
				{
					rt.sigils.push_back(
						parse::TySigil{ .type = parse::TySigil::SLICE }
					);
				}
				rt.size = TYPE_RES_PTR_SIZE + TYPE_RES_SIZE_SIZE * 2 * rt.outerSliceDims;
				rt.outerSliceDims = 0;//reset, as we already added the sigils.
			}
			else
				rt.size = TYPE_RES_PTR_SIZE;

			rt.sigils.emplace_back(
				parse::TySigil{.type= parse::TySigil::REF }
			);

			return rt;
		},
		varcase(parse::ExprType::LimPrefixExprV<true>&&)->parse::ResolvedType {
			//TODO
			throw std::runtime_error("TODO: resolve lim-prefix-expr type expressions.");
		},
		varcase(parse::ExprType::Slice&&)->parse::ResolvedType {
			//size = {usize ptr}+ {usize len,usize stride}*sliceDims.
			//TODO
			throw std::runtime_error("TODO: resolve slice type expressions.");
		},
		varcase(parse::ExprType::TableV<true>&&) {
			parse::StructRawType& res = *(new parse::StructRawType());
			res.name = lang::MpItmIdV<true>::newEmpty();
			res.fieldNames.reserve(var.size());
			res.fields.reserve(var.size());
			res.fieldOffsets.reserve(var.size());
			size_t idx = 1;
			size_t fieldOffset = 0;
			for (parse::FieldV<true>& field : var)
			{
				ezmatch(std::move(field))(
				ezcase(parse::FieldType::ExprV<true>&& fi) {
					res.fieldNames.emplace_back("0x" + parse::u64ToStr(idx++));
					res.fields.emplace_back(resolveTypeExpr(mpDb, std::move(fi)));
				},
				ezcase(parse::FieldType::Name2ExprV<true>&& fi) {
					res.fieldNames.emplace_back(fi.idx.asSv(mpDb));
					res.fields.emplace_back(resolveTypeExpr(mpDb, std::move(fi.v)));
				},
				ezcase(parse::FieldType::Expr2ExprV<true>&& fi) {
					throw std::runtime_error("FieldType::Expr2ExprV type resolution: TODO not implemented: jit the expression");
				},
				ezcase(const parse::FieldType::NONE _) {
					throw std::runtime_error("FieldType::NONE should not exist in struct type expression.");
				}
				);
				//TODO: fix field offsets changing depending on named field order, and tuple field order between them too.
				if (fieldOffset!=SIZE_MAX)
					continue;

				const parse::ResolvedType& resField = res.fields.back();

				if (resField.size == 0)
				{
					res.fieldOffsets.push_back(SIZE_MAX);//undefined value.
					continue;
				}
				if (!resField.isComplete())
				{
					fieldOffset = SIZE_MAX;
					continue;
				}

				res.fieldOffsets.push_back((fieldOffset + 7)& (~0b111));//ceil to byte boundary.
				fieldOffset += resField.size;
			}

			size_t resSize = parse::ResolvedType::INCOMPLETE_MARK;
			bool resSizeInBits = false;
			if (fieldOffset != SIZE_MAX)
				resSize = fieldOffset;

			return parse::ResolvedType{
				.base = parse::RawTypeKind::Struct(&res),
				.size = resSize
			};
		},
		varcase(parse::ExprType::Union&&)->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve Union type expressions.");
		},


		varcase(parse::ExprType::IfCondV<true>&&) ->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve if type expressions.");
		},


#define Slu_INVALID_EXPR_CASE(_MSG,...) varcase(__VA_ARGS__ &&)->parse::ResolvedType { \
		throw std::runtime_error("Invalid slu " _MSG ", index:" #__VA_ARGS__ "."); \
	}

		Slu_INVALID_EXPR_CASE("expression",parse::ExprType::True),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::False),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::Nil),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::VarArgs),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::PatTypePrefix),
		Slu_INVALID_EXPR_CASE("type", parse::ExprType::TraitExpr),
		Slu_INVALID_EXPR_CASE("type", parse::ExprType::FunctionV<true>),
		Slu_INVALID_EXPR_CASE("type", parse::ExprType::Lifetime),
		varcase(parse::ExprType::MultiOpV<true>&&)->parse::ResolvedType {
			throw std::runtime_error("Multi-op type expressions are ment to be desuagared before type resolution.");
		});
#undef Slu_INVALID_EXPR_CASE

		return resTy;
	}
}