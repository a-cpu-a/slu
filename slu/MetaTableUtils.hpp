/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <vector>
#include <slu/Include.hpp>

#include <slu/WrapFunc.hpp>
#include <slu/types/complex/TableKey.hpp>


namespace slu
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

		// The function should be (/*const*/ THIS_T& thisObject, const slu::TableKey& key), /*const*/ -> optionally const
		// Call it as if this was a method inside MetaTableGetters (var.Slu_newGetter("fn", abc);)
#define Slu_getter(_NAME,_CPP_FUNC) \
	{_NAME, Slu_wrapRaw(_NAME, _CPP_FUNC),true}



		// Lets you add a method
#define Slu_method(_NAME,_CPP_FUNC) Slu_wrap(_NAME,_CPP_FUNC)

	};

	inline int handleMetatableGet(lua_State* L, const MetaTableGetters& getters)
	{
		if (lua_gettop(L) != 2)
			return slu::lua_error(L,
				"Getters must have " 
				LUACC_NUMBER "2 " LC_arguments 
				" (thisObject, key)");

		if (!slu::TableKey::check(L, 2))
			return slu::lua_error(L, LC_invalid " getter key");

		const slu::TableKey key = slu::TableKey::read(L, 2);

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

		throw slu::Error(
			"Unknown key in getter " 
			LUACC_START_SINGLE_STRING "{}" LUACC_STRING_SINGLE "'",
			strKey
		);
	}
#define Slu_setupGetHandler(_GETTERS) \
	{ "__index", \
		[](lua_State* L){ return slu::handleMetatableGet(L,_GETTERS); } \
	}

	// Combine with ```lua_setfield(L, -2, "__index");```, to add to a metatable.
#define Slu_pushGetHandler(L,_GETTERS) \
	lua_pushcfunction(L, \
		[](lua_State* L){ return slu::handleMetatableGet(L,_GETTERS); } \
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

		// The function should be (/*const*/ THIS_T& thisObject, const slu::TableKey& key, /*const*/ VAL_T& val), /*const*/ -> optionally const
#define Slu_setter(_NAME,_CPP_FUNC) Slu_wrap(_NAME,_CPP_FUNC)
	};

	inline int handleMetatableSet(lua_State* L, const MetaTableSetters& setters)
	{
		if (lua_gettop(L) != 3)
			return slu::lua_error(L,
				"Setters must have " 
				LUACC_NUMBER "3 " LC_arguments
				" (thisObject, key, newVal)");

		if (!slu::TableKey::check(L, 2))
			return slu::lua_error(L, LC_invalid " setter key");

		const slu::TableKey key = slu::TableKey::read(L, 2);

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

		throw slu::Error(
			"Unknown key in setter " 
			LUACC_START_SINGLE_STRING "{}" LUACC_STRING_SINGLE "'",
			strKey
		);
	}
#define Slu_setupSetHandler(_SETTERS) \
	{ "__newindex", \
		[](lua_State* L){ return slu::handleMetatableSet(L,_SETTERS); } \
	}

	// Combine with ```lua_setfield(L, -2, "__newindex");```, to add to a metatable.
#define Slu_pushSetHandler(L,_SETTERS) \
	lua_pushcfunction(L, \
		[](lua_State* L){ return slu::handleMetatableSet(L,_SETTERS); } \
	)

}


#define Slu_makeGetter(_THIS_OBJ_ARG,_RET_VAL) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slu::TableKey& key) \
	{ \
		return _RET_VAL; \
	})


//Checks if _CHECK is true, and if not, will throw a error
#define Slu_makeGetterChecking(_THIS_OBJ_ARG,_CHECK,_ERROR,_RET_VAL) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slu::TableKey& key) \
	{ \
		if(!(_CHECK)) \
			throw slu::Error(_ERROR); \
		return _RET_VAL; \
	})



#define Slu_makeSetter(_THIS_OBJ_ARG,_VAL_OBJ_ARG,_SET_CODE) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slu::TableKey& key,_VAL_OBJ_ARG) \
	{ \
		_SET_CODE; \
		return slu::Void(); \
	})

//Checks if _CHECK is true, and if not, will throw a error
#define Slu_makeSetterChecking(_THIS_OBJ_ARG,_VAL_OBJ_ARG,_CHECK,_ERROR,_SET_CODE) \
	/* wrap inside + to turn into function pointer */ (+[](_THIS_OBJ_ARG, const slu::TableKey& key,_VAL_OBJ_ARG) \
	{ \
		if(!(_CHECK)) \
			throw slu::Error(_ERROR); \
		_SET_CODE; \
		return slu::Void(); \
	})


#define Slu_setterBuilder(_FIELD_NAME,    _FIELD_OBJ,_THIS_OBJ_T,_FIELD_OBJ_T,    _CHECK,_ERROR) \
	Slu_setter(#_FIELD_NAME, \
		Slu_makeSetterChecking(\
			const _THIS_OBJ_T & thisObj, \
			const decltype(_FIELD_OBJ_T ::_FIELD_NAME)& val, \
			_CHECK, \
			_ERROR, \
			_FIELD_OBJ._FIELD_NAME = val \
		) \
	)


// _FIELD_VAL_WRAPPER can be empty, so you get "... ,, ..."
#define Slu_getterBuilder(_FIELD_NAME,_FIELD_VAL_WRAPPER,    _FIELD_OBJ,_THIS_OBJ_T,    _CHECK,_ERROR) \
	Slu_getter(#_FIELD_NAME, \
		Slu_makeGetterChecking(\
			const _THIS_OBJ_T & thisObj, \
			_CHECK, \
			_ERROR, \
			_FIELD_VAL_WRAPPER (_FIELD_OBJ._FIELD_NAME) \
		) \
	)