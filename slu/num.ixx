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
export module slu.num;

namespace slu
{
	export template<bool pad = false>
	constexpr std::string u64ToStr(const uint64_t v)
	{
		std::string res;
		for (size_t i = 0; i < 16; i++)
		{
			const uint64_t va = uint64_t(v) >> (60 - 4 * i);

			if constexpr (!pad)
			{
				if (va == 0)
					continue;
			}
			const uint8_t c = va & 0xF;
			if (c <= 9)
				res += ('0' + c);
			else
				res += ('A' + (c - 10));
		}
		return res;
	}
	export constexpr std::string u128ToStr(const uint64_t lo, const uint64_t hi)
	{
		std::string res = "0x";
		if (hi != 0)
		{
			res += u64ToStr(hi);
			res += u64ToStr<true>(lo);
		} else if (lo != 0)
			res += u64ToStr(lo);
		else
			res += '0';
		return res;
	}
} //namespace slu