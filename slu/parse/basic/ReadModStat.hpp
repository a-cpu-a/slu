/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

import slu.ast.state;
import slu.parse.input;
import slu.parse.com.skip_space;
import slu.parse.com.token;
#include <slu/parse/adv/ReadName.hpp>

namespace slu::parse
{
	template<AnyInput In>
	inline bool readModStat(In& in, const ast::Position place, const lang::ExportData exported)
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
			}
			else
			{
				in.genData.addStat(place, StatType::Mod<In>{ modName, exported });
			}

			return true;
		}
		return false;
	}
}