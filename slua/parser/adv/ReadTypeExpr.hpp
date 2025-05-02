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

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/ReadExprBase.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/adv/ReadExpr.hpp>
#include <slua/parser/adv/ReadTable.hpp>
#include <slua/parser/adv/ReadTraitExpr.hpp>
#include <slua/parser/errors/CharErrors.h>
#include <slua/parser/errors/KwErrors.h>

namespace slua::parse
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
			ret.unOps.push_back(uOp);
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
			in.skip();
			ret.data = TypeExprDataType::SLICER(std::make_unique<Expression<In>>(readExpr(in,false)));
			requireToken(in, "]");
			break;
		case '{': // table constructor
			ret.data = TypeExprDataType::TABLE_CONSTRUCTOR(readTableConstructor(in,false));
			break;
		case 's'://safe fn
			if (!checkReadTextToken(in, "safe"))
				break;
			requireToken(in,"fn");
			ret.data = readFnType(in, OptSafety::SAFE);
			break;
		case 'u'://unsafe fn
			if (!checkReadTextToken(in, "unsafe"))
				break;
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