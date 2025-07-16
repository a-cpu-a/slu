/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/parser/BasicGenData.hpp>

namespace slu::mlvl
{
	inline const size_t TYPE_RES_SIZE_SIZE = 64;//TODO: unhardcode this, allow 8 bits too.
	inline const size_t TYPE_RES_PTR_SIZE = 64;//TODO: unhardcode this, allow 8 bits too.

	parse::ResolvedType resolveTypeExprFromExpr(parse::BasicMpDb mpDb, parse::ExpressionV<true>&& type)
	{
		//TODO
	}
	parse::ResolvedType resolveTypeExpr(parse::BasicMpDb mpDb, parse::TypeExpr&& type)
	{
		//TODO: change mlir conv code to use global itm's.
		parse::ResolvedType resTy = ezmatch(std::move(type.data))(
		varcase(const parse::TypeExprDataType::ERR_INFERR) {
			return parse::ResolvedType::getInferred();
		},
		varcase(parse::TypeExprDataType::LITERAL_STRING&&){
			return parse::ResolvedType::getConstType(parse::RawTypeKind::String{std::move(var.v)});
		},
		varcase(const parse::TypeExprDataType::NUMERAL_I64){
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Int64{ var.v });
		},
		varcase(const parse::TypeExprDataType::NUMERAL_U64){
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint64{ var.v });
		},
		varcase(const parse::TypeExprDataType::NUMERAL_I128) {
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Int128{ var.lo,var.hi });
		},
		varcase(const parse::TypeExprDataType::NUMERAL_U128) {
			return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
		},
		varcase(parse::TypeExprDataType::FN&&) {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve FN type expressions.");
		},
		varcase(parse::TypeExprDataType::DYN&&) {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve DYN type expressions.");
		},
		varcase(parse::TypeExprDataType::IMPL&&) {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve IMPL type expressions.");
		},
		varcase(parse::TypeExprDataType::ERR&&) {
			//TODO replace with ?~~var equivelant.
			throw std::runtime_error("TODO: resolve ERR type expressions.");
		},

		varcase(parse::TypeExprDataType::FUNC_CALL&&) {
			//TODO: resolve basic ops, jit all else.
			if(var.argChain.size() != 1)
				throw std::runtime_error("TODO: resolve complex FUNC_CALL type expressions.");
		},
		varcase(parse::TypeExprDataType::LIM_PREFIX_EXP&&) {
			//TODO
		},
		varcase(parse::TypeExprDataType::SLICER&&) {
			//size = {usize ptr}+ {usize len,usize stride}*sliceDims.
			//TODO
		},
		varcase(parse::TypeExprDataType::Struct&&) {
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
				ezcase(parse::FieldType::EXPRv<true>&& fi) {
					res.fieldNames.emplace_back("0x" + parse::u64ToStr(idx++));
					res.fields.emplace_back(resolveTypeExprFromExpr(mpDb, std::move(fi.v)));
				},
				ezcase(parse::FieldType::NAME2EXPRv<true>&& fi) {
					res.fieldNames.emplace_back(fi.idx.asSv(mpDb));
					res.fields.emplace_back(resolveTypeExprFromExpr(mpDb, std::move(fi.v)));
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
		varcase(parse::TypeExprDataType::Union&&) {
			//TODO
			//return parse::ResolvedType::getConstType(parse::RawTypeKind::Uint128{ var.lo,var.hi });
			throw std::runtime_error("TODO: resolve Union type expressions.");
		},



		varcase(parse::TypeExprDataType::MULTI_OP&&) {
			throw std::runtime_error("Multi-op type expressions are ment to be desuagared before type resolution.");
		});
		resTy.hasMut = type.hasMut;
	}
}