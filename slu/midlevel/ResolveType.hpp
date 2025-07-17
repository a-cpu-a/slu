/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/parser/BasicGenData.hpp>

namespace slu::mlvl
{
	inline const size_t TYPE_RES_SIZE_SIZE = 64;//TODO: unhardcode this, allow 8 bits too.
	inline const size_t TYPE_RES_PTR_SIZE = 64;//TODO: unhardcode this, allow 8 bits too.

	parse::ResolvedType resolveTypeExpr(parse::BasicMpDb mpDb, parse::ExpressionV<true>&& type)
	{
		//TODO: change mlir conv code to use global itm's.
		parse::ResolvedType resTy = ezmatch(std::move(type.data))(
		varcase(const parse::ExprType::Inferr) {
			return parse::ResolvedType::getInferred();
		},
		varcase(const parse::ExprType::OPEN_RANGE)->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve open range type expressions.");
		},
		varcase(parse::ExprType::LITERAL_STRING&&){
			return parse::ResolvedType::getConstType(parse::RawTypeKind::String{std::move(var.v)});
		},
		varcase(const parse::ExprType::I64){
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Int64{ var.v });
		},
		varcase(const parse::ExprType::NUMERAL_U64){
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint64{ var.v });
		},
		varcase(const parse::ExprType::NUMERAL_I128) {
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Int128{ var.lo,var.hi });
		},
		varcase(const parse::ExprType::NUMERAL_U128) {
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
		},
		varcase(const parse::ExprType::NUMERAL)->parse::ResolvedType {
			throw std::runtime_error("TODO: resolve numeral type expressions.");
		},
		varcase(parse::ExprType::FnType&&)->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve FN type expressions.");
		},
		varcase(parse::ExprType::Dyn&&)->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve DYN type expressions.");
		},
		varcase(parse::ExprType::Impl&&)->parse::ResolvedType {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve IMPL type expressions.");
		},
		varcase(parse::ExprType::Err&&)->parse::ResolvedType {
			//TODO replace with ?~~var equivelant.
			throw std::runtime_error("TODO: resolve ERR type expressions.");
		},

		varcase(parse::ExprType::FUNC_CALLv<true>&&)->parse::ResolvedType {
			//TODO: resolve basic ops, jit all else.
			if(var.argChain.size() != 1)
				throw std::runtime_error("TODO: resolve complex FUNC_CALL type expressions.");

			throw std::runtime_error("TODO: resolve func-call type expressions.");
		},
		varcase(parse::ExprType::LIM_PREFIX_EXPv<true>&&)->parse::ResolvedType {
			//TODO
			throw std::runtime_error("TODO: resolve lim-prefix-expr type expressions.");
		},
		varcase(parse::ExprType::Slice&&)->parse::ResolvedType {
			//size = {usize ptr}+ {usize len,usize stride}*sliceDims.
			//TODO
			throw std::runtime_error("TODO: resolve slice type expressions.");
		},
		varcase(parse::ExprType::TABLE_CONSTRUCTORv<true>&&) {
			parse::StructRawType& res = *(new parse::StructRawType());
			res.name = lang::MpItmIdV<true>::newEmpty();
			res.fieldNames.reserve(var.v.size());
			res.fields.reserve(var.v.size());
			res.fieldOffsets.reserve(var.v.size());
			size_t idx = 1;
			size_t fieldOffset = 0;
			for (parse::FieldV<true>& field : var.v)
			{
				ezmatch(std::move(field))(
				ezcase(parse::FieldType::EXPRv<true>&& fi) {
					res.fieldNames.emplace_back("0x" + parse::u64ToStr(idx++));
					res.fields.emplace_back(resolveTypeExpr(mpDb, std::move(fi.v)));
				},
				ezcase(parse::FieldType::NAME2EXPRv<true>&& fi) {
					res.fieldNames.emplace_back(fi.idx.asSv(mpDb));
					res.fields.emplace_back(resolveTypeExpr(mpDb, std::move(fi.v)));
				},
				ezcase(parse::FieldType::EXPR2EXPRv<true>&& fi) {
					throw std::runtime_error("FieldType::EXPR2EXPRv type resolution: TODO not implemented: jit the expression");
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

				res.fieldOffsets.push_back(fieldOffset);
				fieldOffset += resField.bitSizeByteCeil();
			}

			size_t resSize = parse::ResolvedType::INCOMPLETE_MARK;
			bool resSizeInBits = false;
			if (fieldOffset != SIZE_MAX)
			{
				if (res.fields.size() == 1)
				{
					parse::ResolvedType& field = res.fields[0];
					resSize = field.size;
					resSizeInBits = field.sizeInBits;
				}
				else
					resSize = fieldOffset / 8;
			}

			return parse::ResolvedType{
				.base = parse::RawTypeKind::Struct(&res),
				.size = resSize,
				.sizeInBits = resSizeInBits
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

		Slu_INVALID_EXPR_CASE("expression",parse::ExprType::TRUE),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::FALSE),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::NIL),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::VARARGS),
		Slu_INVALID_EXPR_CASE("expression", parse::ExprType::PAT_TYPE_PREFIX),
		Slu_INVALID_EXPR_CASE("type", parse::ExprType::TRAIT_EXPR),
		Slu_INVALID_EXPR_CASE("type", parse::ExprType::FUNCTION_DEFv<true>),
		Slu_INVALID_EXPR_CASE("type", parse::ExprType::LIFETIME),
		varcase(parse::ExprType::MULTI_OPERATIONv<true>&&)->parse::ResolvedType {
			throw std::runtime_error("Multi-op type expressions are ment to be desuagared before type resolution.");
		});
#undef Slu_INVALID_EXPR_CASE

		return resTy;
	}
}