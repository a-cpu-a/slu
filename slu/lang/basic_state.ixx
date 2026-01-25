/*
    Slu language compiler, a computer program compiler.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;

#include <compare>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
export module slu.lang.basic_state;

namespace slu::lang
{
	export struct MpItmId;

	//Mp's
	export using ModPath = std::vector<std::string>;
	export using ModPathView = std::span<const std::string>;
	export using ViewModPath = std::vector<std::string_view>;
	export using ViewModPathView = std::span<const std::string_view>;
} //namespace slu::lang
extern "C++"
{
namespace slu::parse
{
	struct BasicMpDbData;
	struct BasicMpDb;
	std::string_view _fwdConstructBasicMpDbAsSv(
	    BasicMpDbData* data, lang::MpItmId thiz);
	lang::ViewModPath _fwdConstructBasicMpDbAsVmp(
	    BasicMpDbData* data, lang::MpItmId thiz);
} //namespace slu::parse
}
namespace slu::lang
{
	export struct ModPathId
	{
		size_t id;
		constexpr auto operator<=>(const lang::ModPathId&) const = default;
	};
	export struct LocalObjId
	{
		size_t val;
		constexpr static lang::LocalObjId newEmpty()
		{
			return lang::LocalObjId{SIZE_MAX};
		}

		constexpr auto operator<=>(const lang::LocalObjId&) const = default;
	};

	export struct MpItmId
	{
		lang::LocalObjId id; // Practically a string pool lol
		//SIZE_MAX -> empty
		lang::ModPathId mp;

		static constexpr lang::MpItmId newEmpty()
		{
			return lang::MpItmId{lang::LocalObjId{SIZE_MAX}};
		}

		constexpr bool empty() const
		{
			return id.val == SIZE_MAX;
		}

		std::string_view asSv(const auto& v) const
		{
			return v.asSv({*(const lang::MpItmId*)this});
		}
		std::string_view asSv(const parse::BasicMpDbData& v) const
		{
			return parse::_fwdConstructBasicMpDbAsSv(
			    const_cast<parse::BasicMpDbData*>(&v), {*this});
		}
		ViewModPath asVmp(const auto& v) const
		{
			return v.asVmp(*this);
		}
		ViewModPath asVmp(const parse::BasicMpDbData& v) const
		{
			return parse::_fwdConstructBasicMpDbAsVmp(
			    const_cast<parse::BasicMpDbData*>(&v), {*this});
		}

		constexpr auto operator<=>(const lang::MpItmId&) const = default;
	};
	export using PoolString = lang::LocalObjId; //implicitly unknown mp.

	//Might in the future also contain data about other stuff, like export
	//control (crate,self,tests,...).
	export using ExportData = bool;

	export template<class T>
	concept AnyMp
	    = std::same_as<T, lang::ModPathView> || std::same_as<T, ViewModPathView>
	    || std::same_as<T, ViewModPath> || std::same_as<T, lang::ModPath>;

	export struct HashModPathView
	{
		using is_transparent = void;
		template<AnyMp T> constexpr std::size_t operator()(const T& data) const
		{
			std::size_t seed
			    = data.size(); // Start with size to add some variation

			std::hash<typename T::value_type> hasher;

			for (const auto& str : data)
			{
				seed ^= hasher(str) * 31 + (seed << 6)
				    + (seed >> 2); // Someone, fix this lol
			}

			return seed;
		}
	};
	export struct EqualModPathView
	{
		using is_transparent = void;
		constexpr bool operator()(
		    const AnyMp auto& a, const AnyMp auto& b) const
		{
			return std::equal(
			    std::begin(a), std::end(a), std::begin(b), std::end(b));
		}
	};
} //namespace slu::lang