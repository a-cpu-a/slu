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

namespace slu::parse
{
	//Doesnt check (, starts after it too.
	template<bool isLocal, AnyInput In>
	inline ParamList<In> readParamList(In& in)
	{
		ParamList<In> res;
		skipSpace(in);

		if(in.peek()==')')
		{
			in.skip();
			return res;
		}

		res.emplace_back(readFuncParam<isLocal>(in));

		while (checkReadToken(in, ","))
			res.emplace_back(readFuncParam<isLocal>(in));

		requireToken(in, ")");

		return res;
	}
	template<class T,bool structOnly,AnyInput In>
	inline void readStructStat(In& in, const Position place, const ExportData exported)
	{
		in.genData.pushLocalScope();
		T res{};
		res.exported = exported;

		res.name = in.genData.addLocalObj(readName(in));

		skipSpace(in);
		if (in.peek() == '(')
		{
			in.skip();

			res.params = readParamList<true>(in);
			skipSpace(in);
		}
		requireToken(in, "{");
		res.type = readTable<false>(in, false);
		res.local2Mp = in.genData.popLocalScope();

		in.genData.addStat(place, std::move(res));
	}
}