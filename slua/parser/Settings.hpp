/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

namespace sluaParse
{
#define _Slua_MAKE_SETTING_FUNC(_NAME) \
	consteval bool _NAME() const \
	{ \
		bool r = false; \
		((r |= SettingTs::_NAME()),...); \
		return r; \
	}

	template<class THIS, class... SettingTs>
	struct Setting : SettingTs...
	{
		_Slua_MAKE_SETTING_FUNC(spacedFuncCallStrForm);

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
	};
#undef _Slua_MAKE_SETTING_FUNC


#define _Slua_MAKE_SETTING_CVAR(_NAME) \
	struct C_ ## _NAME : Setting<C_ ## _NAME> \
	{ \
		consteval bool _NAME() const {return true;} \
	}; \
	inline constexpr auto _NAME = C_ ## _NAME()


	_Slua_MAKE_SETTING_CVAR(spacedFuncCallStrForm);


#undef _Slua_MAKE_SETTING_CVAR
}
