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
#include <format>
#include <span>
#include <string>
#include <vector>

export module slu.parse.vec_input;
import slu.settings;
import slu.ast.mp_data;
import slu.ast.pos;
import slu.parse.input;

namespace slu::parse
{
	export template<AnySettings SettingsT = Setting<void>> struct VecInput
	{
		constexpr VecInput(SettingsT) {}
		constexpr VecInput() = default;

		constexpr static SettingsT settings()
		{
			return SettingsT();
		}

		BasicGenData genData;

		std::vector<std::string> handledErrors;

		std::string_view fName;
		size_t curLine = 1;
		size_t curLinePos = 0;

		std::span<const uint8_t> text;
		size_t idx = 0;

		void restart()
		{
			curLine = 1;
			curLinePos = 0;
			idx = 0;
		}

		uint8_t peek()
		{
			if (idx >= text.size())
				throw EndOfStreamError(*this);

			return text[idx];
		}

		//offset 0 is the same as peek()
		uint8_t peekAt(const size_t offset)
		{
			if (idx > SIZE_MAX - offset || //idx + count overflows, so...
			    idx + offset >= text.size())
			{
				throw EndOfStreamError(*this);
			}

			return text[idx + offset];
		}

		// span must be valid until next get(), so, any
		// other peek()'s must not invalidate these!!!
		std::span<const uint8_t> peek(const size_t count)
		{
			if (idx > SIZE_MAX - count ||  //idx + count overflows, so...
			    idx + count > text.size()) //position after this peek() can be
			                               // at text.size(), but not above it
			{
				throw EndOfStreamError(*this);
			}

			std::span<const uint8_t> res = text.subspan(idx, count);

			return res;
		}
		void skip(const size_t count = 1)
		{
			//if (idx >= text.size())
			//	throw EndOfStreamError();

			curLinePos += count;
			idx += count;
		}

		uint8_t get()
		{
			if (idx >= text.size())
				throw EndOfStreamError(*this);

			curLinePos++;
			return text[idx++];
		}
		// span must be valid until next get(), so, any
		// peek()'s must not invalidate these!!!
		std::span<const uint8_t> get(const size_t count)
		{
			if (idx > SIZE_MAX - count ||  //idx + count overflows, so...
			    idx + count > text.size()) //position after this get() can be at
			                               // text.size(), but not above it
			{
				throw EndOfStreamError(*this);
			}

			std::span<const uint8_t> res = text.subspan(idx, count);

			idx += count;
			curLinePos += count;

			return res;
		}

		/* Returns true, while stream still has stuff */
		operator bool() const
		{
			return idx < text.size();
		}
		//Passing 0 is the same as (!in)
		bool isOob(const size_t offset) const
		{
			return (idx > SIZE_MAX - offset || idx + offset >= text.size());
		}


		//Error output

		std::string_view fileName() const
		{
			return fName;
		}
		ast::Position getLoc() const
		{
			return {curLine, curLinePos};
		}
		void newLine()
		{
			curLine++;
			curLinePos = 0;
		}


		void handleError(const std::string e)
		{
			handledErrors.push_back(e);
		}
		bool hasError() const
		{
			return !handledErrors.empty();
		}
	};
} //namespace slu::parse
