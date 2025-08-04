/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <span>
#include <vector>
#include <memory>
#include <variant>

#include <slu/ext/CppMatch.hpp>
#include <slu/ext/ExtendVariant.hpp>

namespace slu
{
	template<bool SIGNED, bool NEGATIVIZED = false>
	struct Integer128
	{
		uint64_t lo = 0;
		uint64_t hi = 0;

		constexpr static Integer128<true, false> fromInt(const int64_t val)
		{
			Integer128<true, false> o{};
			o.lo = static_cast<uint64_t>(val);
			o.hi = val < 0 ? ~0ULL : 0;
			return o;
		}
		constexpr static Integer128<false, false> fromInt(const uint64_t val)
		{
			Integer128<false, false> o{};
			o.lo = val;
			o.hi = 0;
			return o;
		}

		constexpr Integer128<SIGNED, !NEGATIVIZED> negative() const {
			return { lo,hi };
		}
		constexpr Integer128<false, false> abs() const {
			if constexpr (SIGNED)
			{
				if (hiSigned() < 0)
				{
					return (Integer128<false, false>{ ~lo, ~hi })
						+ Integer128<false, false>{1, 0};
				}
			}
			return { lo, hi };
		}

		constexpr bool isNegative() const {
			if constexpr (SIGNED)
				return (static_cast<int64_t>(hi) < 0) != NEGATIVIZED;//!= is xor
			else
				return NEGATIVIZED;
		}
		constexpr auto hiSigned() const {
			if constexpr (SIGNED)
				return static_cast<int64_t>(hi);
			else
				return static_cast<uint64_t>(hi);
		}
		template<bool O_SIGN, bool O_NEGATIVIZED>
		constexpr std::strong_ordering operator<=>(const Integer128<O_SIGN, O_NEGATIVIZED>& o) const {
			if constexpr (SIGNED || NEGATIVIZED || O_SIGN || O_NEGATIVIZED)
			{
				const bool neg = isNegative();
				const bool oNeg = o.isNegative();
				if (neg && oNeg)
					return o.negative() <=> negative();
				if (neg)
					return std::strong_ordering::less; // this is negative, o is positive
				if (oNeg)
					return std::strong_ordering::greater; // this is positive, o is negative }
			}
			//Both are positive... after negativization that is.

			if constexpr (NEGATIVIZED || O_NEGATIVIZED)
				return abs() <=> o.abs();

			if constexpr (SIGNED == O_SIGN)
			{
				// Both signed or both unsigned
				if (hi != o.hi) return hiSigned() <=> o.hiSigned();
				return lo <=> o.lo;
			}
			else if constexpr (SIGNED)
			{
				// this is signed, o is unsigned
				if (isNegative()) return std::strong_ordering::less;
				// Now 'this' is positive. Compare magnitudes directly.
				if (hi != o.hi) return hi <=> o.hi;
				return lo <=> o.lo;
			}
			else
			{
				// this is unsigned, o is signed
				if (o.isNegative()) return std::strong_ordering::greater;
				// Now 'o' is positive. Compare magnitudes directly.
				if (hi != o.hi) return hi <=> o.hi;
				return lo <=> o.lo;
			}
		}
		constexpr auto operator<=>(const int64_t val) const {
			return *this <=> fromInt(val);
		}
		constexpr auto operator<=>(const uint64_t val) const {
			return *this <=> fromInt(val);
		}

		template<bool O_SIGN, bool O_NEGATIVIZED>
		constexpr bool operator==(const Integer128<O_SIGN, O_NEGATIVIZED>& o) const {
			if (lo == 0 && o.lo == 0 && hi == 0 && o.hi == 0)
				return true;
			if (isNegative() != o.isNegative())
				return false;
			return lo == o.lo && hi == o.hi;
		}
		constexpr bool operator==(const int64_t val) const {
			return *this == fromInt(val);
		}
		constexpr bool operator==(const uint64_t val) const {
			return *this == fromInt(val);
		}

		template<bool O_NEGATIVIZED>
		constexpr Integer128 operator+(Integer128<SIGNED, O_NEGATIVIZED> o) const requires(!NEGATIVIZED && (!SIGNED || !O_NEGATIVIZED))
		{
			if constexpr (O_NEGATIVIZED)
			{
				o.lo = ~o.lo;
				o.hi = ~o.hi;
				auto s = Integer128<false, false>{ .lo = o.lo,.hi = o.hi } + fromInt(1ULL); // Negate;
				o.lo = s.lo;
				o.hi = s.hi;
			}
			Integer128 res = *this;
			res.lo += o.lo;
			res.hi += o.hi;
			if (res.lo < lo)//overflow check
				res.hi++;
			return res;
		}
		constexpr Integer128 operator-(const Integer128& o) const requires(!NEGATIVIZED) {
			return *this + o.negative();
		}

		constexpr size_t bitWidth() const requires(!SIGNED && !NEGATIVIZED)
		{
			if (hi == 0)
				return std::bit_width(lo);
			return 64 + std::bit_width(hi);
		}
		template<bool O_NEGATIVIZED>
		constexpr bool lteOtherPlus1(const Integer128<false, O_NEGATIVIZED>& o) const requires(!SIGNED)
		{
			if constexpr (NEGATIVIZED && O_NEGATIVIZED)
			{//-A <= -B+1 (->) A >= B-1.
				if (o == 0ULL) return true;
				return negative() >= (o.negative() + fromInt(UINT64_MAX));//-1
			}
			else if constexpr (NEGATIVIZED && !O_NEGATIVIZED)
			{//-A <= B+1
				if (o.lo == UINT64_MAX && o.hi == UINT64_MAX)
					return true;
				return *this <= (o + fromInt(1ULL));
			}
			else if constexpr (!NEGATIVIZED && O_NEGATIVIZED)
			{//A <= -B+1 (->) -A >= B-1.
				if (o == 0ULL) return *this <= 1ULL;
				return negative() >= (o.negative() + fromInt(UINT64_MAX));//-1
			}
			else
			{//A <= B+1.
				if (o.lo == UINT64_MAX && o.hi == UINT64_MAX)
					return true;
				return *this <= (o + fromInt(1ULL));
			}
		}
		constexpr Integer128 shift(uint8_t count) const requires(!NEGATIVIZED)
		{
			Integer128 res = *this;
			res.hi <<= count;
			res.hi |= lo >> (64 - count);
			res.lo <<= count;
			return res;
		}
	};
}