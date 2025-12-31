module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>
#include <unordered_set>

export module slu.parse.basic.label;

import slu.ast.pos;
import slu.ast.state;
import slu.lang.basic_state;
import slu.parse.input;
import slu.parse.com.name;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	export template<AnyInput In>
	void readLabel(In& in, const ast::Position place)
	{
		//label ::= ‘::’ Name ‘::’
		//SL label ::= ‘:::’ Name ‘:’

		requireToken(in, ":::");

		if (checkReadTextToken(in, "unsafe"))
		{
			requireToken(in, ":");
			in.genData.setUnsafe();
			return in.genData.addStat(place, StatType::UnsafeLabel{});
		} else if (checkReadTextToken(in, "safe"))
		{
			requireToken(in, ":");
			in.genData.setSafe();
			return in.genData.addStat(place, StatType::SafeLabel{});
		}

		const lang::MpItmId res = in.genData.addLocalObj(readName(in));

		requireToken(in, ":");

		return in.genData.addStat(place, parse::StatType::Label<In>{res});
	}
} //namespace slu::parse