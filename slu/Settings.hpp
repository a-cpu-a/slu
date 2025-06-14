﻿/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <type_traits>

namespace slu::parse
{

	template<class THIS, class... SettingTs>
	struct Setting : SettingTs...
	{
		template<class... OSettingTs>
		consteval auto operator|(const Setting<OSettingTs...>& o) const
		{
			// Create a new Setting that combines all base classes.
			Setting<SettingTs..., OSettingTs...> result{ static_cast<const SettingTs&>(*this)..., static_cast<const OSettingTs&>(o)... };
			return result;
		}
		template<class T>
		consteval auto operator|(const T& o) const
		{
			if constexpr (std::is_same_v<THIS, void>)
			{
				// Create a new Setting that combines all base classes.
				Setting<void, SettingTs..., T> result(
					static_cast<const SettingTs&>(*this)...,
					static_cast<const T&>(o)
				);
				return result;
			}
			else
			{
				// Create a new Setting that combines all base classes.
				Setting<void, THIS, SettingTs..., T> result(
					static_cast<const THIS&>(*this),
					static_cast<const SettingTs&>(*this)...,
					static_cast<const T&>(o)
				);
				return result;
			}
		}
		template<class T>
		consteval bool operator&(const T& o) const
		{
			return o.isOn(*this);
		}
	};
#undef _Slu_MAKE_SETTING_FUNC

	template<class T>
	struct _AnySetting_impl
	{
		using v = typename T::isSetting;
	};
	template<class T2, class... T3>
	struct _AnySetting_impl<Setting<T2, T3...>>
	{
		using v = std::true_type;
	};

	template<class T>
	concept AnySettings = _AnySetting_impl<std::remove_cvref_t<T>>::v::value;

#define _Slu_MAKE_SETTING_CVAR(_NAME) \
	struct _C_ ## _NAME : Setting<_C_ ## _NAME> \
	{ \
		using isSetting = std::true_type; \
		template<class THIS, class... SettingTs> \
		consteval bool isOn(Setting<THIS,SettingTs...> settings) const \
		{ \
			bool r = false; \
			((r |= std::is_same_v<SettingTs,_C_ ## _NAME>),...); \
			return r; \
		} \
	}; \
	inline constexpr auto _NAME = _C_ ## _NAME()

	//Parser only:
	_Slu_MAKE_SETTING_CVAR(spacedFuncCallStrForm);
	_Slu_MAKE_SETTING_CVAR(noIntOverflow);
	_Slu_MAKE_SETTING_CVAR(sluSyn);
	_Slu_MAKE_SETTING_CVAR(numberSpacing); // stuff like: 100_100

	constexpr auto sluCommon = spacedFuncCallStrForm
		| noIntOverflow 
		| sluSyn
		| numberSpacing;

#undef _Slu_MAKE_SETTING_CVAR

#ifdef Slu_NoConcepts
#define AnyCfgable class
#else
	template<class T>
	concept AnyCfgable = requires(T t) {
		{ t.settings() } -> AnySettings;
	};
#endif
}
