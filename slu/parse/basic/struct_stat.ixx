module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>

export module slu.parse.basic.struct_stat;

import slu.ast.state;
import slu.parse.input;
import slu.parse.com.name;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	//Doesnt check (, starts after it too.
	export template<bool isLocal, AnyInput In>
	ParamList<isLocal> readParamList(In& in)
	{
		ParamList<isLocal> res;
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
	export template<class T,bool structOnly,AnyInput In>
	void readStructStat(In& in, const ast::Position place, const lang::ExportData exported)
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