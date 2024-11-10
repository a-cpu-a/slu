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


		MetaTableGetters& _add(const char* name, lua_CFunction func, bool doCall = true)
		{
			list.emplace_back(name, func, doCall);
			maxFieldLen = std::max(maxFieldLen, strlen(list.back().name));
			return *this;
		}

		// The function should be (/*const*/ THIS_T& thisObject, const slua::TableKey& key), /*const*/ -> optionally const
		// Call it as if this was a method inside MetaTableGetters (var.SLua_newGetter("fn", abc);)
#define SLua_newGetter(_NAME,_CPP_FUNC) _add(_NAME, SLua_WrapRaw(_NAME, _CPP_FUNC))


		MetaTableGetters& newGetterRaw(const char* name, lua_CFunction cFunc) {
			return _add(name, cFunc);
		}

		// Lets you add a method
		// Call it as if this was a method inside MetaTableGetters (var.SLua_newMethod("fn", abc);)
#define SLua_newMethod(_NAME,_CPP_FUNC) _add(_NAME, SLua_WrapRaw(_NAME, _CPP_FUNC), false)

		// Lets you add a method
		MetaTableGetters& newMethodRaw(const char* name, lua_CFunction cFunc) {
			return _add(name, cFunc, false);
		}
	};

	inline int handleMetatableGet(lua_State* L, const MetaTableGetters& getters)
	{
		if (lua_gettop(L) != 2)
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


		MetaTableSetters& _add(const char* name, lua_CFunction func)
		{
			list.emplace_back(name, func);
			maxFieldLen = std::max(maxFieldLen, strlen(list.back().name));
			return *this;
		}

		// The function should be (/*const*/ THIS_T& thisObject, const slua::TableKey& key, /*const*/ VAL_T& val), /*const*/ -> optionally const
		// Call it as if this was a method inside MetaTableSetters (var.SLua_newSetter("fn", abc);)
#define SLua_newSetter(_NAME,_CPP_FUNC) _add(_NAME, SLua_WrapRaw(_NAME, _CPP_FUNC))

		MetaTableSetters& newSetterRaw(const char* name, lua_CFunction cFunc) {
			return _add(name, cFunc);
		}
	};

	inline int handleMetatableSet(lua_State* L, const MetaTableSetters& setters)
	{
		if (lua_gettop(L) != 2)
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




//Checks if _CHECK is true, and if not, will throw a error
#define SLua_MakeGetterChecking(_THIS_OBJ_ARG,_CHECK,_ERROR,_RET_VAL) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slua::TableKey& key) \
	{ \
		if(!(_CHECK)) \
			throw slua::Error(_ERROR); \
		return _RET_VAL; \
	})