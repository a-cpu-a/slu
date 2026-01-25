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
module;
#include <cstdint>
export module slu.parse.manage_newl;
import slu.parse.input;

namespace slu::parse
{
	export enum class ParseNewlineState : uint8_t
	{
		NONE,
		CARI,
	};

	//Returns if newline was added
	export template<bool skipPreNl>
	bool manageNewlineState(
	    const char ch, ParseNewlineState& nlState, parse::AnyInput auto& in)
	{
		switch (nlState)
		{
		case slu::parse::ParseNewlineState::NONE:
			if (ch == '\n')
			{
				if constexpr (skipPreNl)
					in.skip();
				in.newLine();
				return true;
			} else if (ch == '\r')
				nlState = slu::parse::ParseNewlineState::CARI;
			break;
		case slu::parse::ParseNewlineState::CARI:
			if (ch != '\r')
			{ // \r\n, or \r(normal char)
				if constexpr (skipPreNl)
					in.skip();
				in.newLine();
				nlState = slu::parse::ParseNewlineState::NONE;
				return true;
			} else // \r\r
			{
				if constexpr (skipPreNl)
					in.skip();
				in.newLine();
				return true;
			}
			break;
		}
		return false;
	}
} //namespace slu::parse