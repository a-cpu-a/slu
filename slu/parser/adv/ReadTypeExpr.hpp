/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>
#include <memory>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/ReadExprBase.hpp>
#include <slu/parser/adv/RequireToken.hpp>
#include <slu/parser/adv/ReadExpr.hpp>
#include <slu/parser/adv/ReadTable.hpp>
#include <slu/parser/adv/ReadTraitExpr.hpp>
#include <slu/parser/errors/CharErrors.h>
#include <slu/parser/errors/KwErrors.h>

namespace slu::parse
{
	//No space skip!
	//YOU parse the 'fn'
	template<AnyInput In>
	inline TypeExprData readFnType(In& in, const OptSafety safety)
	{// [safety] "fn" typeExp "->" typeExp
		TypeExprDataType::FN res{};
		res.safety = safety;

		res.argType = std::make_unique<TypeExpr>(readTypeExpr(in, false));
		requireToken(in, "->");
		res.retType = std::make_unique<TypeExpr>(readTypeExpr(in, false));

		return res;
	}
	//No space skip!
	template<AnyInput In>
	inline TypeExpr readTypeExpr(In& in,const bool allowMut,const bool readBiOp=true)
	{
		const Position startPos = in.getLoc();

		TypeExpr ret;
		ret.place = startPos;
		bool intentionalyErrInferr = false;

		if (allowMut && checkReadTextToken(in, "mut"))
		{
			ret.hasMut = true;
			skipSpace(in);
		}

		while (true)
		{
			const UnOpType uOp = readOptUnOp(in,true);
			if (uOp == UnOpType::NONE)break;
			if (uOp == UnOpType::TO_REF)
			{
				ret.unOps.push_back(readToRefLifetimes(in, uOp));
				continue;
			}
			ret.unOps.push_back({ .type = uOp });
		}

		const char firstChar = in.peek();

		switch (firstChar)
		{
		case '?':
			in.skip();
			ret.data = TypeExprDataType::ERR_INFERR{};
			intentionalyErrInferr = true;
			break;
		case '/':
			requireToken(in, "//");
			ret.data = TypeExprDataType::ERR{std::make_unique<TypeExpr>(readTypeExpr(in,allowMut))};
			break;
		case '[':// [exp]
		{
			in.skip();
			TypeExprDataType::SLICER res = std::make_unique<Expression<In>>(readExpr(in, false));
			ret.data = std::move(res);
			requireToken(in, "]");
			break;
		}
		case '{': // table constructor
			ret.data = TypeExprDataType::Struct(readTableConstructor(in,false));
			break;
		case 's'://safe fn struct
			if (!checkReadTextToken(in, "safe"))
			{
				if(checkReadTextToken(in, "struct"))
					ret.data = TypeExprDataType::Struct(readTableConstructor(in, false));
				break;
			}
			requireToken(in,"fn");
			ret.data = readFnType(in, OptSafety::SAFE);
			break;
		case 'u'://unsafe fn union
			if (!checkReadTextToken(in, "unsafe"))
			{
				if (checkReadTextToken(in, "union"))
					ret.data = TypeExprDataType::Union(readTableConstructor(in, false));
				break;
			}
			requireToken(in,"fn");
			ret.data = readFnType(in, OptSafety::UNSAFE);
			break;
		case 'f'://fn
			if (checkReadTextToken(in, "fn"))
				ret.data = readFnType(in, OptSafety::DEFAULT);
			break;
		case 'd':
			if (checkReadTextToken(in, "dyn"))
			{
				ret.data = TypeExprDataType::DYN(readTraitExpr(in));
			}
			break;
		case 'i':
			if (checkReadTextToken(in, "impl"))
			{
				ret.data = TypeExprDataType::IMPL(readTraitExpr(in));
			}
			break;
		case 't':
		{
			if (checkReadTextToken(in, "trait"))
				ret.data = TypeExprDataType::TRAIT_TY{};
			break;
		}
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
			ret.data = readNumeral<TypeExprData,false>(in, firstChar);
			break;
		default:
			break;
		}


		if (!intentionalyErrInferr 
			&& std::holds_alternative<TypeExprDataType::ERR_INFERR>(ret.data))
		{
			if (firstChar != '(' && !isValidNameStartChar(firstChar))
				throwExpectedTypeExpr(in);

			ret.data = parsePrefixExprVar<TypeExprData, true,true>(in, false, firstChar);
		}

		if (!readBiOp)return ret;

		skipSpace(in);

		if (!in)return ret;

		const BinOpType firstBinOp = readOptBinOp(in);

		if (firstBinOp == BinOpType::NONE)
			return ret;

		TypeExprDataType::MULTI_OP resData{};

		resData.first = std::make_unique<TypeExpr>(std::move(ret));
		resData.extra.emplace_back(firstBinOp, readTypeExpr(in, allowMut, false));

		while (true)
		{
			skipSpace(in);

			if (!in) break;//File ended

			const BinOpType binOp = readOptBinOp(in);

			if (binOp == BinOpType::NONE) break;

			resData.extra.emplace_back(binOp, readTypeExpr(in, allowMut, false));
		}

		ret.place = startPos;
		ret.data = std::move(resData);
		ret.hasMut = false;

		return ret;
	}
}