/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

#include <slu/parser/State.hpp>
#include <slu/parser/Input.hpp>
#include <slu/parser/adv/SkipSpace.hpp>
#include <slu/parser/adv/RequireToken.hpp>
#include <slu/parser/adv/ReadName.hpp>

namespace slua::parse
{
	//Doesnt check (, starts after it too.
	template<AnyInput In>
	inline ParamList<In> readParamList(In& in)
	{
		ParamList<In> res;
		skipSpace(in);

		if(in.peek()==')')
		{
			in.skip();
			return res;
		}

		res.emplace_back(readFuncParam(in));

		while (checkReadToken(in, ","))
			res.emplace_back(readFuncParam(in));

		requireToken(in, ")");

		return res;
	}
	template<class T,bool structOnly,AnyInput In>
	inline void readStructStat(In& in, const Position place, const ExportData exported)
	{
		T res{};
		res.exported = exported;

		res.name = in.genData.resolveUnknown(readName(in));

		skipSpace(in);
		if (in.peek() == '(')
		{
			in.skip();

			res.params = readParamList(in);
		}
		if constexpr (structOnly)
			res.type = readTableConstructor(in, false);
		else
		{
			switch (in.peek())
			{
			case '{':
				res.type = readTypeExpr(in, false);
				break;
			case '=':
				in.skip();
				res.type = readTypeExpr(in, false);
				break;
			default:
				throwExpectedStructOrAssign(in);
			}
		}

		in.genData.addStat(place, std::move(res));
	}
}