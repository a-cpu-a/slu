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
		bool doCall = false;// If true, then func will be called, to get the return value & push stuff to lua
	};

	// Allows you to create methods, fields
	struct MetaTableGetters
	{
		std::vector<MetaTableGetter> list;
		size_t maxFieldLen = 0;

		MetaTableGetters() {}
		MetaTableGetters(const std::initializer_list<MetaTableGetter>& setters)
		{
			list = setters;
			for (const MetaTableGetter& setter : list)
				maxFieldLen = std::max(maxFieldLen, strlen(setter.name));
		}

		// The function should be (/*const*/ THIS_T& thisObject, const slua::TableKey& key), /*const*/ -> optionally const
		// Call it as if this was a method inside MetaTableGetters (var.SLua_newGetter("fn", abc);)
#define SLua_Getter(_NAME,_CPP_FUNC) {_NAME, SLua_WrapRaw(_NAME, _CPP_FUNC),true}



		// Lets you add a method
#define SLua_Method(_NAME,_CPP_FUNC) SLua_Wrap(_NAME,_CPP_FUNC)

	};

	inline int handleMetatableGet(lua_State* L, const MetaTableGetters& getters)
	{
		if (lua_gettop(L) != 2)
			return slua::lua_error(L, "Getters must have " LUACC_NUMBER "2 " LC_arguments " (thisObject, key)");

		if (!slua::TableKey::check(L, 2))
			return slua::lua_error(L, LUACC_INVALID "Invalid" LUACC_DEFAULT " getter key");

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

		throw slua::Error("Unknown key in getter " LUACC_START_SINGLE_STRING + strKey + LUACC_STRING_SINGLE "'");
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

		MetaTableSetters() {}
		MetaTableSetters(const std::initializer_list<MetaTableSetter>& setters)
		{
			list = setters;
			for (const MetaTableSetter& setter : list)
				maxFieldLen = std::max(maxFieldLen, strlen(setter.name));
		}

		// The function should be (/*const*/ THIS_T& thisObject, const slua::TableKey& key, /*const*/ VAL_T& val), /*const*/ -> optionally const
#define SLua_Setter(_NAME,_CPP_FUNC) SLua_Wrap(_NAME,_CPP_FUNC)
	};

	inline int handleMetatableSet(lua_State* L, const MetaTableSetters& setters)
	{
		if (lua_gettop(L) != 3)
			return slua::lua_error(L, "Setters must have " LUACC_NUMBER "3 " LC_arguments " (thisObject, key, newVal)");

		if (!slua::TableKey::check(L, 2))
			return slua::lua_error(L, LUACC_INVALID "Invalid" LUACC_DEFAULT " setter key");

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

		throw slua::Error("Unknown key in setter " LUACC_START_SINGLE_STRING + strKey + LUACC_STRING_SINGLE "'");
	}
#define SLua_SetupSetHandler(_SETTERS) {"__newindex", [](lua_State* L){ return slua::handleMetatableSet(L,_SETTERS); } }

}


#define SLua_MakeGetter(_THIS_OBJ_ARG,_RET_VAL) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slua::TableKey& key) \
	{ \
		return _RET_VAL; \
	})


//Checks if _CHECK is true, and if not, will throw a error
#define SLua_MakeGetterChecking(_THIS_OBJ_ARG,_CHECK,_ERROR,_RET_VAL) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slua::TableKey& key) \
	{ \
		if(!(_CHECK)) \
			throw slua::Error(_ERROR); \
		return _RET_VAL; \
	})



#define SLua_MakeSetter(_THIS_OBJ_ARG,_VAL_OBJ_ARG,_SET_CODE) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slua::TableKey& key,_VAL_OBJ_ARG) \
	{ \
		_SET_CODE; \
		return slua::Void(); \
	})

//Checks if _CHECK is true, and if not, will throw a error
#define SLua_MakeSetterChecking(_THIS_OBJ_ARG,_VAL_OBJ_ARG,_CHECK,_ERROR,_SET_CODE) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slua::TableKey& key,_VAL_OBJ_ARG) \
	{ \
		if(!(_CHECK)) \
			throw slua::Error(_ERROR); \
		_SET_CODE; \
		return slua::Void(); \
	})


#define SLua_SetterBuilder(_FIELD_NAME,    _FIELD_OBJ,_THIS_OBJ_T,_FIELD_OBJ_T,    _CHECK,_ERROR) \
	SLua_Setter(#_FIELD_NAME, \
		SLua_MakeSetterChecking(\
			const _THIS_OBJ_T & thisObj, \
			const decltype(_FIELD_OBJ_T ::_FIELD_NAME)& val, \
			_CHECK, \
			_ERROR, \
			_FIELD_OBJ._FIELD_NAME = val \
		) \
	)


// _FIELD_VAL_WRAPPER can be empty, so you get "... ,, ..."
#define SLua_GetterBuilder(_FIELD_NAME,_FIELD_VAL_WRAPPER,    _FIELD_OBJ,_THIS_OBJ_T,    _CHECK,_ERROR) \
	SLua_Getter(#_FIELD_NAME, \
		SLua_MakeGetterChecking(\
			const _THIS_OBJ_T & thisObj, \
			_CHECK, \
			_ERROR, \
			_FIELD_VAL_WRAPPER (_FIELD_OBJ._FIELD_NAME) \
		) \
	)