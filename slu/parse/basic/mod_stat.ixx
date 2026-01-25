/*
    Slu language compiler, a computer program compiler.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;
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