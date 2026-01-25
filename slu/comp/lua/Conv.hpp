/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
#pragma once

#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

import slu.comp.cfg;
import slu.comp.conv_data;

namespace slu::comp::lua
{
	template<typename T>
	concept AnyIgnoredStat = std::same_as<T, parse::StatType::GotoV<true>>
	    || std::same_as<T, parse::StatType::Semicol>
	    || std::same_as<T, parse::StatType::Use>
	    || std::same_as<T, parse::StatType::FnDeclV<true>>
	    || std::same_as<T, parse::StatType::FunctionDeclV<true>>
	    || std::same_as<T,
	        parse::StatType::ExternBlockV<true>> //ignore, as desugaring will
	                                             //remove it
	    || std::same_as<T,
	        parse::StatType::UnsafeBlockV<true>> //ignore, as desugaring will
	                                             //remove it
	    || std::same_as<T, parse::StatType::DropV<true>>
	    || std::same_as<T, parse::StatType::ModV<true>>
	    || std::same_as<T, parse::StatType::ModAsV<true>>
	    || std::same_as<T, parse::StatType::UnsafeLabel>
	    || std::same_as<T, parse::StatType::SafeLabel>;

	struct ConvData : CommonConvData
	{
		parse::LuaMpDb& luaDb;
		parse::Output<>& out;
	};

	inline void convStat(const ConvData& conv)
	{
		ezmatch(conv.stat.data)(

		    varcase(const auto&){},


		    //Ignore these
		    varcase(const AnyIgnoredStat auto&){});
	}
	inline void conv(const ConvData& conv)
	{
		//TODO: Implement the conversion logic here
	}
} //namespace slu::comp::lua