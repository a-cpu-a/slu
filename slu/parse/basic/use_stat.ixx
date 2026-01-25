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
#include <utility>

export module slu.parse.basic.use_stat;

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
	bool readUseStat(
	    In& in, const ast::Position place, const lang::ExportData exported)
	{
		if (checkReadTextToken(in, "use"))
		{
			StatType::Use res{};
			res.exported = exported;

			bool root = false;
			skipSpace(in);
			if (in.peek() == ':')
			{
				requireToken<false>(in, ":>"); //Modpath root
				requireToken(in, "::");
				root = true;
			}

			lang::ModPath mp = readModPath(in); //Moved @ IMPORT
			res.base = root ? in.genData.resolveRootName(mp)
			                : in.genData.resolveName(mp);

			if (in.peek() == ':')
			{
				if (in.peekAt(1) == ':' && in.peekAt(2) == '*')
				{
					in.skip(3);
					res.useVariant = UseVariantType::EVERYTHING_INSIDE{};
				} else if (in.peekAt(1) == ':' && in.peekAt(2) == '{')
				{
					in.skip(3);
					UseVariantType::LIST_OF_STUFF list;
					list.push_back(in.genData.addLocalObj(
					    readName<NameCatagory::MP_START>(in)));
					while (checkReadToken(in, ","))
					{
						list.push_back(in.genData.addLocalObj(
						    readName<NameCatagory::MP_START>(in)));
					}
					requireToken(in, "}");
					res.useVariant = std::move(list);
				} else
				{ // Neither, prob just no semicol
					res.useVariant = UseVariantType::IMPORT{
					    in.genData.addLocalObj(std::move(mp.back()))};
				}
			} else
			{
				if (checkReadTextToken(in, "as"))
				{
					res.useVariant = UseVariantType::AS_NAME{
					    in.genData.addLocalObj(readName(in))};
				} else
				{ // Prob just no semicol
					res.useVariant = UseVariantType::IMPORT{
					    in.genData.addLocalObj(std::move(mp.back()))};
				}
			}
			in.genData.addStat(place, std::move(res));
			return true;
		}
		return false;
	}
} //namespace slu::parse