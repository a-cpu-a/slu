/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <vector>
#include <slua/Include.hpp>

#include <slua/WrapFunc.hpp>
#include <slua/types/complex/TableKey.hpp>


namespace slua
{
	struct MetaTableGetter
	{
		const char* name;

		// Could return a function, or a value, or... a error
		lua_CFunction func;//maybe wrapped, maybe custom
		bool doCall = true;// If true, then func will be called, to get the return value & push stuff to lua
	};

	// Allows you to create methods, fields
	struct MetaTableGetters
	{
		std::vector<MetaTableGetter> list;
		size_t maxFieldLen = 0;
	};

	template<typename RET_T, typename THIS_T>
	using MetaTableGetterFunc = RET_T(*)(THIS_T& objectThis, const slua::TableKey&);

	template<typename RET_T, typename THIS_T>
	inline MetaTableGetter newGetter(const char* name, MetaTableGetterFunc<RET_T, THIS_T> cppFunc) {
		return MetaTableGetter(name, SLua_WrapRaw(cppFunc));
	}
	// Lets you add a method
	inline MetaTableGetter newMethod(const char* name, auto cppFunc) {
		return MetaTableGetter(name, SLua_WrapRaw(cppFunc), true);
	}
	// Lets you add a method
	inline MetaTableGetter newMethod(const char* name, lua_CFunction cFunc) {
		return MetaTableGetter(name, cFunc, true);
	}

	inline int handleMetatableGet(lua_State* L, const MetaTableGetters& getters)
	{
		if (lua_gettop(_L) != 2)
			return slua::error(L, "Getters must have " LUACC_NUMBER "2" LUACC_ARGUMENT " arguments" LUACC_DEFAULT " (thisObject, key)");

		if (!slua::TableKey::check(L, 2))
			return slua::error(L, LUACC_INVALID "Invalid" LUACC_DEFAULT " getter key");

		const slua::TableKey key = slua::TableKey::read(L, 2);

		const std::string strKey = key.toString();

		if (strKey.size() <= getters.maxFieldLen)
		{
			//find it, if possible

			for (const MetaTableGetter& getter : getters.list)
			{
				if (getter.name != strKey)
					continue;

				if (!getter.doCall)
				{
					lua_pushcfunction(L, getter.func);
					return 1;//1 function
				}

				return getter.func(L);
			}
		}

		return slua::error(L, "Unknown key in getter " LUACC_STRING_SINGLE "'" LUACC_STRING_INNER + strKey + LUACC_STRING_SINGLE "'");
	}
#define SLua_SetupGetHandler(_GETTERS) {"__index", [](lua_State* L){ return slua::handleMetatableGet(L,_GETTERS); } }



	struct MetaTableSetter
	{
		const char* name;

		lua_CFunction func;//maybe wrapped, maybe custom
	};


	// Allows you to create fields that can be changed
	struct MetaTableSetters
	{
		std::vector<MetaTableSetter> list;
		size_t maxFieldLen = 0;
	};

	template<typename RET_T, typename THIS_T, typename VAL_T>
	using MetaTableSetterFunc = RET_T(*)(THIS_T& objectThis, const slua::TableKey&, VAL_T& val);

	template<typename RET_T, typename THIS_T, typename VAL_T>
	inline MetaTableSetter newSetter(const char* name, MetaTableSetterFunc<RET_T, THIS_T, VAL_T> cppFunc) {
		return MetaTableSetter(name, SLua_WrapRaw(cppFunc));
	}
	inline MetaTableGetter newSetter(const char* name, lua_CFunction cFunc) {
		return MetaTableGetter(name, cFunc);
	}

	inline int handleMetatableSet(lua_State* L, const MetaTableSetters& setters)
	{
		if (lua_gettop(_L) != 2)
			return slua::error(L, "Setters must have " LUACC_NUMBER "3" LUACC_ARGUMENT " arguments" LUACC_DEFAULT " (thisObject, key, newVal)");

		if (!slua::TableKey::check(L, 2))
			return slua::error(L, LUACC_INVALID "Invalid" LUACC_DEFAULT " setter key");

		const slua::TableKey key = slua::TableKey::read(L, 2);

		const std::string strKey = key.toString();

		if (strKey.size() <= setters.maxFieldLen)
		{
			//find it, if possible

			for (const MetaTableSetter& getter : setters.list)
			{
				if (getter.name != strKey)
					continue;

				return getter.func(L);
			}
		}

		return slua::error(L, "Unknown key in setter " LUACC_STRING_SINGLE "'" LUACC_STRING_INNER + strKey + LUACC_STRING_SINGLE "'");
	}
#define SLua_SetupSetHandler(_SETTERS) {"__newindex", [](lua_State* L){ return slua::handleMetatableSet(L,_SETTERS); } } }

}