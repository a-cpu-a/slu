module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>
#include <unordered_set>
#include <memory>

#include <slu/parse/adv/ReadExprBase.hpp>
export module slu.parse.basic.trait_expr;

import slu.ast.state;
import slu.parse.input;
import slu.parse.com.skip_space;
import slu.parse.com.tok;

namespace slu::parse
{
	export template<AnyInput In>
	TraitExpr readTraitExpr(In& in)
	{
		skipSpace(in);
		TraitExpr ret;
		ret.place = in.getLoc();

		ret.traitCombo.emplace_back(readBasicExpr(in, false, false));

		while (in)
		{
			skipSpace(in);
			if (in.peek() != '+')
				break;
			in.skip();

			ret.traitCombo.emplace_back(readBasicExpr(in, false, false));
		}
		return ret;
	}
}