/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <span>
namespace slu::lang
{
	template<bool isSlu>
	struct MpItmIdV;
}
namespace slu::parse
{
	struct BasicMpDbData;
	struct BasicMpDb;
	std::string_view _fwdConstructBasicMpDbAsSv(BasicMpDbData* data, lang::MpItmIdV<true> thiz);
}
namespace slu::lang
{
	//Mp refs

	using ModPath = std::vector<std::string>;
	using ModPathView = std::span<const std::string>;
	using ViewModPath = std::vector<std::string_view>;
	using ViewModPathView = std::span<const std::string_view>;

	struct ModPathId
	{
		size_t id; //Id 0 -> unknownRoot

		constexpr auto operator<=>(const ModPathId&)const = default;
	};
	struct LocalObjId
	{
		size_t val;

		constexpr auto operator<=>(const LocalObjId&)const = default;
	};


	template<bool isSlu>
	struct MpItmIdV;

	template<bool isSlu>
	struct MpItmIdCommonV
	{
		LocalObjId id;// Practically a string pool lol
		//SIZE_MAX -> empty

		static constexpr MpItmIdV<isSlu> newEmpty() {
			return MpItmIdV<isSlu>{ LocalObjId{SIZE_MAX} };
		}

		constexpr bool empty() const {
			return id.val == SIZE_MAX;
		}
		std::string_view asSv(const auto& v) const {
			return v.asSv({ *this });
		}

		std::string_view asSv(const parse::BasicMpDbData& v) const requires(isSlu) {
			return parse::_fwdConstructBasicMpDbAsSv(const_cast<parse::BasicMpDbData*>(&v), { *this });
		}

		constexpr auto operator<=>(const MpItmIdCommonV&)const = default;
	};
	
	template<bool isSlu>
	struct MpItmIdV : MpItmIdCommonV<false>
	{
		using MpItmIdCommonV<false>::newEmpty;

		constexpr auto operator<=>(const MpItmIdV&)const = default;
	};
	template<>
	struct MpItmIdV<true> : MpItmIdCommonV<true>
	{
		using MpItmIdCommonV<true>::newEmpty;

		ViewModPath asVmp(const auto& v) const {
			return v.asVmp(*this);
		}
		ModPathId mp;

		constexpr auto operator<=>(const MpItmIdV<true>&)const = default;
	};

	//Might in the future also contain data about other stuff, like export control (crate,self,tests,...).
	using ExportData = bool;

template<class T>
	concept AnyMp = 
		std::same_as<T, ModPathView>
		|| std::same_as<T, ViewModPathView>
		|| std::same_as<T, ModPathView>
		|| std::same_as<T, ModPath>;


	struct HashModPathView
	{
		using is_transparent = void;
		template<AnyMp T>
		constexpr std::size_t operator()(const T& data) const {
			std::size_t seed = data.size();  // Start with size to add some variation

			std::hash<typename T::value_type> hasher;

			for (const auto& str : data)
			{
				seed ^= hasher(str) * 31 + (seed << 6) + (seed >> 2); // Someone, fix this lol
			}

			return seed;
		}
	};

	struct EqualModPathView
	{
		using is_transparent = void;
		constexpr bool operator()(const AnyMp auto& lhs, const AnyMp auto& rhs)const {
			return std::equal(begin(lhs), end(lhs),
				begin(rhs), end(rhs));
		}
	};
}