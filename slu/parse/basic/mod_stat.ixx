module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>
#include <string>
#include <utility>

export module slu.parse.basic.mod_stat;

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
	bool readModStat(
	    In& in, const ast::Position place, const lang::ExportData exported)
	{
		if (checkReadTextToken(in, "mod"))
		{
			std::string name = readName(in);
			const lang::MpItmId modName = in.genData.addLocalObj(name);

			if (checkReadTextToken(in, "as"))
			{
				StatType::ModAs<In> res{};
				res.exported = exported;
				res.name = modName;

				requireToken(in, "{");
				in.genData.pushScope(in.getLoc(), std::move(name));
				res.code = readGlobStatList<false>(in);
				requireToken(in, "}");

				in.genData.addStat(place, std::move(res));
			} else
			{
				in.genData.addStat(place, StatType::Mod<In>{modName, exported});
			}

			return true;
		}
		return false;
	}
} //namespace slu::parse