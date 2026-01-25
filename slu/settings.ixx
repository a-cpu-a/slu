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
#include <cstdint>
#include <type_traits>

export module slu.settings;

namespace slu::parse
{
	export template<class THIS, class... SettingTs>
	struct Setting : SettingTs...
	{
		constexpr static Setting settings()
		{
			return Setting();
		}

		template<class... OSettingTs>
		consteval auto operator|(const Setting<OSettingTs...>& o) const
		{
			// Create a new Setting that combines all base classes.
			Setting<SettingTs..., OSettingTs...> result{
			    static_cast<const SettingTs&>(*this)...,
			    static_cast<const OSettingTs&>(o)...};
			return result;
		}
		template<class T> consteval auto operator|(const T& o) const
		{
			if constexpr (std::is_same_v<THIS, void>)
			{
				// Create a new Setting that combines all base classes.
				Setting<void, SettingTs..., T> result(
				    static_cast<const SettingTs&>(*this)...,
				    static_cast<const T&>(o));
				return result;
			} else
			{
				// Create a new Setting that combines all base classes.
				Setting<void, THIS, SettingTs..., T> result(
				    static_cast<const THIS&>(*this),
				    static_cast<const SettingTs&>(*this)...,
				    static_cast<const T&>(o));
				return result;
			}
		}
		template<class T> consteval bool operator&(const T& o) const
		{
			return o.isOn(*this);
		}
	};
#undef _Slu_MAKE_SETTING_FUNC

	template<class T> struct _AnySetting_impl
	{
		using v = typename T::isSetting;
	};
	template<class T2, class... T3> struct _AnySetting_impl<Setting<T2, T3...>>
	{
		using v = std::true_type;
	};

	export template<class T>
	concept AnySettings = _AnySetting_impl<std::remove_cvref_t<T>>::v::value;

#define _Slu_MAKE_SETTING_CVAR(_NAME)                                   \
	struct _C_##_NAME : Setting<_C_##_NAME>                             \
	{                                                                   \
		using isSetting = std::true_type;                               \
		template<class THIS, class... SettingTs>                        \
		consteval bool isOn(Setting<THIS, SettingTs...> settings) const \
		{                                                               \
			bool r = false;                                             \
			((r |= std::is_same_v<SettingTs, _C_##_NAME>), ...);        \
			return r;                                                   \
		}                                                               \
	};                                                                  \
	export constexpr auto _NAME = _C_##_NAME()

	//Parser only:
	_Slu_MAKE_SETTING_CVAR(noIntOverflow);
	_Slu_MAKE_SETTING_CVAR(numberSpacing); // stuff like: 100_100

	export constexpr auto sluCommon = noIntOverflow | numberSpacing;

#undef _Slu_MAKE_SETTING_CVAR

#ifdef Slu_NoConcepts
#define AnyCfgable class
#else
	export template<class T>
	concept AnyCfgable = requires(T t) {
		{ t.settings() } -> AnySettings;
	};
#endif
} //namespace slu::parse
