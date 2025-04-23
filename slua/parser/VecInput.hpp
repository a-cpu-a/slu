/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <format>
#include <vector>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/Settings.hpp>
#include "Position.hpp"
#include "Input.hpp"

namespace slua::parse
{
	template<AnySettings SettingsT = Setting<void>>
	struct VecInput
	{
		constexpr VecInput(SettingsT) {}
		constexpr VecInput() = default;

		constexpr static SettingsT settings()
		{
			return SettingsT();
		}

		std::vector<std::string> handledErrors;

		std::string fName;
		size_t curLine = 1;
		size_t curLinePos = 0;

		std::span<const uint8_t> text;
		size_t idx = 0;

		uint8_t peek()
		{
			if (idx >= text.size())
				throw EndOfStreamError(*this);

			return text[idx];
		}

		//offset 0 is the same as peek()
		uint8_t peekAt(const size_t offset)
		{
			if (idx > SIZE_MAX - offset ||	//idx + count overflows, so...
				idx + offset >= text.size())
			{
				throw EndOfStreamError(*this);
			}

			return text[idx + offset];
		}

		// span must be valid until next get(), so, any other peek()'s must not invalidate these!!!
		std::span<const uint8_t> peek(const size_t count)
		{
			if (idx > SIZE_MAX - count ||	//idx + count overflows, so...
				idx + count > text.size())	//position after this peek() can be at text.size(), but not above it
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
		// span must be valid until next get(), so, any peek()'s must not invalidate these!!!
		std::span<const uint8_t> get(const size_t count)
		{
			if (idx > SIZE_MAX - count ||	//idx + count overflows, so...
				idx + count > text.size())	//position after this get() can be at text.size(), but not above it
			{
				throw EndOfStreamError(*this);
			}

			std::span<const uint8_t> res = text.subspan(idx, count);

			idx += count;
			curLinePos += count;

			return res;
		}

		/* Returns true, while stream still has stuff */
		operator bool() const {
			return idx < text.size();
		}
		//Passing 0 is the same as (!in)
		bool isOob(const size_t offset) const {
			return (
				idx > SIZE_MAX - offset ||
				idx + offset >= text.size()
				);
		}


		//Error output

		std::string fileName() const {
			return fName;
		}
		Position getLoc() const {
			return { curLine,curLinePos };
		}
		void newLine() {
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
}