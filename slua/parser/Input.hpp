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

		uint8_t peek()
		{
			if (idx >= text.size())
				throw EndOfStreamError();

			return text[idx];
		}

		//offset 0 is the same as peek()
		uint8_t peekAt(const size_t offset)
		{
			if (idx + offset >= text.size()
				|| idx > SIZE_MAX - offset)	//idx + offset overflows, so...
				throw EndOfStreamError();

			return text[idx + offset];
		}

		// span must be valid until next get(), so, any other peek()'s must not invalidate these!!!
		std::span<const uint8_t> peek(const size_t count)
		{
			if (idx + count > text.size()	//position after this peek() can be at text.size(), but not above it
				|| idx > SIZE_MAX - count)	//idx + count overflows, so...
				throw EndOfStreamError();

			std::span<const uint8_t> res = text.subspan(idx, count);

			return res;
		}
		void skip(const size_t count = 1)
		{
			//if (idx >= text.size())
			//	throw EndOfStreamError();

			idx+= count;
		}

		uint8_t get()
		{
			if (idx >= text.size())
				throw EndOfStreamError();

			return text[idx++];
		}
		// span must be valid until next get(), so, any peek()'s must not invalidate these!!!
		std::span<const uint8_t> get(const size_t count)
		{
			if (idx + count > text.size()	//position after this get() can be at text.size(), but not above it
				|| idx > SIZE_MAX - count)	//idx + count overflows, so...
				throw EndOfStreamError();

			std::span<const uint8_t> res = text.subspan(idx, count);

			idx += count;

			return res;
		}

		/* Returns true, while stream still has stuff */
		bool checkEndOfStream() const {
			return idx < text.size();
		}
	};

	//Here, so streamed inputs can be made
	template<class T = Input>
	concept AnyInput = requires(T t) {
		{ t.skip() } -> std::same_as<void>;
		{ t.skip((size_t)100) } -> std::same_as<void>;

		{ t.get() } -> std::same_as<uint8_t>;
		{ t.get((size_t)100) } -> std::same_as<std::span<const uint8_t>>;

		{ t.peek() } -> std::same_as<uint8_t>;
		{ t.peekAt((size_t)100) } -> std::same_as<uint8_t>;
		{ t.peek((size_t)100) } -> std::same_as<std::span<const uint8_t>>;

		/* Returns true, while stream still has stuff */
		{ t.checkEndOfStream() } -> std::same_as<bool>;
	};
}