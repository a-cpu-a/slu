/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <vector>
#include <slu/Include.hpp>

#include <slu/WrapFunc.hpp>
#include <slu/types/complex/TableKey.hpp>


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
#define Slua_getter(_NAME,_CPP_FUNC) \
	{_NAME, Slua_wrapRaw(_NAME, _CPP_FUNC),true}



		// Lets you add a method
#define Slua_method(_NAME,_CPP_FUNC) Slua_wrap(_NAME,_CPP_FUNC)

	};

	inline int handleMetatableGet(lua_State* L, const MetaTableGetters& getters)
	{
		if (lua_gettop(L) != 2)
			return slua::lua_error(L,
				"Getters must have " 
				LUACC_NUMBER "2 " LC_arguments 
				" (thisObject, key)");

		if (!slua::TableKey::check(L, 2))
			return slua::lua_error(L, LC_invalid " getter key");

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

		throw slua::Error(
			"Unknown key in getter " 
			LUACC_START_SINGLE_STRING "{}" LUACC_STRING_SINGLE "'",
			strKey
		);
	}
#define Slua_setupGetHandler(_GETTERS) \
	{ "__index", \
		[](lua_State* L){ return slua::handleMetatableGet(L,_GETTERS); } \
	}

	// Combine with ```lua_setfield(L, -2, "__index");```, to add to a metatable.
#define Slua_pushGetHandler(L,_GETTERS) \
	lua_pushcfunction(L, \
		[](lua_State* L){ return slua::handleMetatableGet(L,_GETTERS); } \
	)



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
#define Slua_setter(_NAME,_CPP_FUNC) Slua_wrap(_NAME,_CPP_FUNC)
	};

	inline int handleMetatableSet(lua_State* L, const MetaTableSetters& setters)
	{
		if (lua_gettop(L) != 3)
			return slua::lua_error(L,
				"Setters must have " 
				LUACC_NUMBER "3 " LC_arguments
				" (thisObject, key, newVal)");

		if (!slua::TableKey::check(L, 2))
			return slua::lua_error(L, LC_invalid " setter key");

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

		throw slua::Error(
			"Unknown key in setter " 
			LUACC_START_SINGLE_STRING "{}" LUACC_STRING_SINGLE "'",
			strKey
		);
	}
#define Slua_setupSetHandler(_SETTERS) \
	{ "__newindex", \
		[](lua_State* L){ return slua::handleMetatableSet(L,_SETTERS); } \
	}

	// Combine with ```lua_setfield(L, -2, "__newindex");```, to add to a metatable.
#define Slua_pushSetHandler(L,_SETTERS) \
	lua_pushcfunction(L, \
		[](lua_State* L){ return slua::handleMetatableSet(L,_SETTERS); } \
	)

}


#define Slua_makeGetter(_THIS_OBJ_ARG,_RET_VAL) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slua::TableKey& key) \
	{ \
		return _RET_VAL; \
	})


//Checks if _CHECK is true, and if not, will throw a error
#define Slua_makeGetterChecking(_THIS_OBJ_ARG,_CHECK,_ERROR,_RET_VAL) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slua::TableKey& key) \
	{ \
		if(!(_CHECK)) \
			throw slua::Error(_ERROR); \
		return _RET_VAL; \
	})



#define Slua_makeSetter(_THIS_OBJ_ARG,_VAL_OBJ_ARG,_SET_CODE) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slua::TableKey& key,_VAL_OBJ_ARG) \
	{ \
		_SET_CODE; \
		return slua::Void(); \
	})

//Checks if _CHECK is true, and if not, will throw a error
#define Slua_makeSetterChecking(_THIS_OBJ_ARG,_VAL_OBJ_ARG,_CHECK,_ERROR,_SET_CODE) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slua::TableKey& key,_VAL_OBJ_ARG) \
	{ \
		if(!(_CHECK)) \
			throw slua::Error(_ERROR); \
		_SET_CODE; \
		return slua::Void(); \
	})


#define Slua_setterBuilder(_FIELD_NAME,    _FIELD_OBJ,_THIS_OBJ_T,_FIELD_OBJ_T,    _CHECK,_ERROR) \
	Slua_setter(#_FIELD_NAME, \
		Slua_makeSetterChecking(\
			const _THIS_OBJ_T & thisObj, \
			const decltype(_FIELD_OBJ_T ::_FIELD_NAME)& val, \
			_CHECK, \
			_ERROR, \
			_FIELD_OBJ._FIELD_NAME = val \
		) \
	)


// _FIELD_VAL_WRAPPER can be empty, so you get "... ,, ..."
#define Slua_getterBuilder(_FIELD_NAME,_FIELD_VAL_WRAPPER,    _FIELD_OBJ,_THIS_OBJ_T,    _CHECK,_ERROR) \
	Slua_getter(#_FIELD_NAME, \
		Slua_makeGetterChecking(\
			const _THIS_OBJ_T & thisObj, \
			_CHECK, \
			_ERROR, \
			_FIELD_VAL_WRAPPER (_FIELD_OBJ._FIELD_NAME) \
		) \
	)