/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <span>

namespace slu::lang
{
	//Mp refs

	using ModPath = std::vector<std::string>;
	using ModPathView = std::span<const std::string>;
	using ViewModPath = std::vector<std::string_view>;

	struct ModPathId
	{
		size_t id; //Id 0 -> unknownRoot
	};
	struct LocalObjId
	{
		size_t val;
	};
	template<bool isSlu>
	struct MpItmIdV
	{
		LocalObjId id;// Practically a string pool lol
		//SIZE_MAX -> empty

		constexpr bool empty() const {
			return id.val == SIZE_MAX;
		}
		std::string_view asSv(const /*AnyNameDbOrGenDataV<isSlu>*/ auto& v) const {
			return v.asSv(*this);
		}
		ViewModPath asVmp(const /*AnyNameDbOrGenDataV<isSlu>*/ auto& v) const {
			return v.asVmp(*this);
		}
	};
	template<>
	struct MpItmIdV<true> : MpItmIdV<false>
	{
		ModPathId mp;
	};

	//Might in the future also contain data about other stuff, like export control (crate,self,tests,...).
	using ExportData = bool;

	struct HashModPathView
	{
		using is_transparent = void;
		constexpr std::size_t operator()(const ModPathView data) const {
			std::size_t seed = data.size();  // Start with size to add some variation
			std::hash<std::string> hasher;

			for (const std::string& str : data)
			{
				seed ^= hasher(str) * 31 + (seed << 6) + (seed >> 2); // Someone, fix this lol
			}

			return seed;
		}
	};

	struct EqualModPathView
	{
		using is_transparent = void;
		constexpr bool operator()(const ModPathView lhs, const ModPathView rhs)const {
			return std::equal(begin(lhs), end(lhs),
				begin(rhs), end(rhs));
		}
	};
}