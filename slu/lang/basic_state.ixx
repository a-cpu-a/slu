module;
/*
** See Copyright Notice inside Include.hpp
*/

#include <string>
#include <vector>
#include <compare>
#include <span>
#include <utility>
#include <unordered_map>
export module slu.lang.basic_state;

namespace slu::lang
{
	struct MpItmId;

	//Mp's
	export using ModPath = std::vector<std::string>;
	export using ModPathView = std::span<const std::string>;
	export using ViewModPath = std::vector<std::string_view>;
	export using ViewModPathView = std::span<const std::string_view>;
}
extern "C++" {
namespace slu::parse
{
	struct BasicMpDbData;
	struct BasicMpDb;
	std::string_view _fwdConstructBasicMpDbAsSv(BasicMpDbData* data, lang::MpItmId thiz);
	lang::ViewModPath _fwdConstructBasicMpDbAsVmp(BasicMpDbData* data, lang::MpItmId thiz);
}
}
namespace slu::lang
{
	export struct ModPathId
	{
		size_t id;
		constexpr auto operator<=>(const lang::ModPathId&)const = default;
	};
	export struct LocalObjId
	{
		size_t val;
		constexpr static lang::LocalObjId newEmpty() {
			return lang::LocalObjId{ SIZE_MAX };
		}

		constexpr auto operator<=>(const lang::LocalObjId&)const = default;
	};
	
	export struct MpItmId
	{
		lang::LocalObjId id;// Practically a string pool lol
		//SIZE_MAX -> empty
		lang::ModPathId mp;

		static constexpr lang::MpItmId newEmpty() {
			return lang::MpItmId{ lang::LocalObjId{ SIZE_MAX } };
		}

		constexpr bool empty() const {
			return id.val == SIZE_MAX;
		}

		std::string_view asSv(const auto& v) const {
			return v.asSv({ *(const lang::MpItmId*)this });
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

		constexpr auto operator<=>(const lang::MpItmId&)const = default;
	};
	export using PoolString = lang::LocalObjId;//implicitly unknown mp.

	//Might in the future also contain data about other stuff, like export control (crate,self,tests,...).
	export using ExportData = bool;

	export template<class T>
	concept AnyMp = 
		std::same_as<T, lang::ModPathView>
		|| std::same_as<T, ViewModPathView>
		|| std::same_as<T, ViewModPath>
		|| std::same_as<T, lang::ModPath>;

	export struct HashModPathView
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
	export struct EqualModPathView
	{
		using is_transparent = void;
		constexpr bool operator()(const AnyMp auto& lhs, const AnyMp auto& rhs)const {
			return std::equal(std::begin(lhs), std::end(lhs),
				std::begin(rhs), std::end(rhs));
		}
	};

	void _test()
	{
		std::unordered_map<lang::ModPath, int, lang::HashModPathView, lang::EqualModPathView> test;
		test[lang::ModPath{ "a", "b", "c" }] = 5;
	}
}