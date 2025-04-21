/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <span>

namespace slua::lang
{
	//Might in the future also contain data about other stuff, like export control (crate,self,tests,...).
	using ExportData = bool;

	using ModPath = std::vector<std::string>;
	using ModPathView = std::span<const std::string>;

	struct UnknownModPathView
	{
		ModPathView v;
	};

	template<class T>
	concept AnyModPathView = std::same_as<T, ModPathView> || std::same_as<T, UnknownModPathView>;

	struct HashModPathView
	{
		using is_transparent = void;
		constexpr std::size_t operator()(ModPathView data) {
			std::size_t seed = data.size();  // Start with size to add some variation
			std::hash<std::string> hasher;

			for (const std::string& str : data)
			{
				seed ^= hasher(str) * 31 + (seed << 6) + (seed >> 2); // Someone, fix this lol
			}

			return seed;
		}
		constexpr std::size_t operator()(const UnknownModPathView data) {
			std::size_t seed = data.v.size()^0xdeadbeef; //not the same as normal mod path view
			std::hash<std::string> hasher;

			for (const std::string& str : data.v)
			{
				seed ^= hasher(str) * 31 + (seed << 6) + (seed >> 2); // Someone, fix this lol
			}

			return seed;
		}
	};

	struct EqualModPathView
	{
		using is_transparent = void;
		constexpr bool operator()(ModPathView lhs, ModPathView rhs) {
			return std::equal(begin(lhs), end(lhs),
				begin(rhs), end(rhs));
		}
		constexpr bool operator()(ModPathView lhs, const UnknownModPathView rhs) {
			return false;
		}
		constexpr bool operator()(const UnknownModPathView lhs, ModPathView rhs) {
			return false;
		}
		constexpr bool operator()(const UnknownModPathView lhs, const UnknownModPathView rhs) {
			return std::equal(begin(lhs.v), end(lhs.v),
				begin(rhs.v), end(rhs.v));
		}
	};
}