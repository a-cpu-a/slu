/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <span>
namespace slu::lang
{
	struct MpItmId;

	//Mp's
	using ModPath = std::vector<std::string>;
	using ModPathView = std::span<const std::string>;
	using ViewModPath = std::vector<std::string_view>;
	using ViewModPathView = std::span<const std::string_view>;
}
namespace slu::parse
{
	struct BasicMpDbData;
	struct BasicMpDb;
	std::string_view _fwdConstructBasicMpDbAsSv(BasicMpDbData* data, lang::MpItmId thiz);
	lang::ViewModPath _fwdConstructBasicMpDbAsVmp(BasicMpDbData* data, lang::MpItmId thiz);
}
namespace slu::lang
{

	struct ModPathId
	{
		size_t id;
		constexpr auto operator<=>(const ModPathId&)const = default;
	};
	struct LocalObjId
	{
		size_t val;
		constexpr static LocalObjId newEmpty() {
			return LocalObjId{ SIZE_MAX };
		}

		constexpr auto operator<=>(const LocalObjId&)const = default;
	};
	
	struct MpItmId
	{
		LocalObjId id;// Practically a string pool lol
		//SIZE_MAX -> empty
		ModPathId mp;

		static constexpr MpItmId newEmpty() {
			return MpItmId{ LocalObjId{ SIZE_MAX } };
		}

		constexpr bool empty() const {
			return id.val == SIZE_MAX;
		}

		std::string_view asSv(const auto& v) const {
			return v.asSv({ *(const MpItmId*)this });
		}
		std::string_view asSv(const parse::BasicMpDbData& v) const {
			return parse::_fwdConstructBasicMpDbAsSv(const_cast<parse::BasicMpDbData*>(&v), { *this });
		}
		ViewModPath asVmp(const auto& v) const {
			return v.asVmp(*this);
		}
		ViewModPath asVmp(const parse::BasicMpDbData& v) const {
			return parse::_fwdConstructBasicMpDbAsVmp(const_cast<parse::BasicMpDbData*>(&v), { *this });
		}

		constexpr auto operator<=>(const MpItmId&)const = default;
	};

	//Might in the future also contain data about other stuff, like export control (crate,self,tests,...).
	using ExportData = bool;

template<class T>
	concept AnyMp = 
		std::same_as<T, ModPathView>
		|| std::same_as<T, ViewModPathView>
		|| std::same_as<T, ViewModPath>
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