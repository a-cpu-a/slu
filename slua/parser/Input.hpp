/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

namespace sluaParse
{
	struct EndOfStreamError : std::exception
	{
		const char* what() const { return "End of stream error"; }
	};

	struct Input
	{
		std::span<const uint8_t> text;
		size_t idx;

		uint8_t get()
		{
			if (idx >= text.size())
				throw EndOfStreamError();

			return text[idx++];
		}
		// span must be valid, until next get()
		std::span<const uint8_t> get(const size_t count)
		{
			if (idx + count > text.size()	//position after this get() can be at text.size(), but not above it
				|| idx > SIZE_MAX - count)	//idx + count overflows, so...
				throw EndOfStreamError();

			std::span<const uint8_t> res = text.subspan(idx, count);

			idx += count;

			return res;
		}

		bool checkEndOfStream() const {
			return idx >= text.size();
		}
	};

	//Here, so streamed inputs can be made
	template<class T = Input>
	concept AnyInput = requires(T t) {
		{ t.get() } -> std::same_as<uint8_t>;
		{ t.get((size_t)100) } -> std::same_as<std::span<const uint8_t>>;
		{ t.checkEndOfStream() } -> std::same_as<bool>;
	};
}