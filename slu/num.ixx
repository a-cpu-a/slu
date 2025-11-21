module;
/*
** See Copyright Notice inside Include.hpp
*/
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