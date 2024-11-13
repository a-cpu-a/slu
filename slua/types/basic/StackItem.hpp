/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <slua/Include.hpp>

#include <slua/Utils.hpp>
#include <slua/types/Converter.hpp>
#include <slua/types/ReadWrite.hpp>

namespace slua
{
	//Turns the item on the top of the stack into a StackItemRef
	struct StackItemRef makeStackItemRef(lua_State* _L, const lua_Integer key);

	//Turns the item on the top of the stack into a StackItemRef
	struct StackItemRef makeStackItemRef(lua_State* _L, const std::string& key);

	// Note: dont mix scopes
	struct StackItem
	{
		lua_State* L;

		const int idx = 0;

		StackItem() {}
		StackItem(lua_State* _L, const int _idx) :L(_L), idx(_idx) {}


		//normal
		StackItem& set(const slua::Pushable auto& newVal, const lua_Integer key)
		{
			slua::push(newVal);
			lua_seti(L, idx, key);
			return *this;
		}
		//normal
		StackItem& set(const slua::Pushable auto& newVal, const std::string& key)
		{
			lua_pushlstring(L, key.data(), key.size());
			slua::push(newVal);
			lua_settable(L, idx, key);
			return *this;
		}

		//normal, but will raw-set by default
		struct StackItemRef get(const lua_Integer key)
		{
			lua_geti(L, idx, key);
			return makeStackItemRef(L, key);
		}
		//normal, but will raw-set by default
		struct StackItemRef get(const std::string& key)
		{
			lua_pushlstring(L, key.data(), key.size());
			lua_gettable(L, idx);
			return makeStackItemRef(L, key);
		}

		//raw
		struct StackItemRef operator[](const lua_Integer key)
		{
			lua_rawgeti(L, idx, key);
			return makeStackItemRef(L, key);
		}
		//raw
		struct StackItemRef operator[](const std::string& key)
		{
			lua_pushlstring(L, key.data(), key.size());
			lua_rawget(L, idx);
			return makeStackItemRef(L, key);
		}

		//normal
		std::string toString() {
			size_t len;
			return std::string(lua_tolstring(L, idx, &len), len);
		}

		lua_Unsigned rawLen() const {
			return lua_rawlen(L, idx);
		}

		//normal
		bool equals(const StackItem& o) {

			return L == o.L && lua_compare(L, idx, o.idx, LUA_OPEQ);
		}
		//raw
		bool operator==(const StackItem& o) const {
			return L == o.L && lua_rawequal(L, idx, o.idx);
		}

		//normal
		bool lessThan(const StackItem& o) {
			return L == o.L && lua_compare(L, idx, o.idx, LUA_OPLT) != 0;
		}
		//"raw"
		bool operator<(const StackItem& o) const {
			if (!lua_isnumber(L, idx) || !lua_isnumber(o.L, idx))
				return false;//cant compare other types

			return lua_tonumber(L, idx) < lua_tonumber(o.L, o.idx);
		}
		//normal
		bool lessThanEqual(const StackItem& o) {
			return L == o.L && lua_compare(L, idx, o.idx, LUA_OPLE) != 0;
		}
		//"raw"
		bool operator<=(const StackItem& o) const {
			if (!lua_isnumber(L, idx) || !lua_isnumber(o.L, idx))
				return false;//cant compare other types

			return lua_tonumber(L, idx) <= lua_tonumber(o.L, o.idx);
		}

		StackItem& setStack(const slua::Pushable auto& newVal)
		{
			slua::push(L, newVal);
			lua_replace(L, idx);// replace this item with the value just pushed

			return *this;
		}

		//raw
		StackItem& operator=(const slua::Pushable auto& newVal) {
			setStack(newVal);
			return *this;
		}

		~StackItem() {
			lua_pop(L, idx);
		}
	};

	struct StackItemRef : StackItem
	{
		const int parentIdx;// -3 -> _G, LUA_REGISTRYINDEX -> registry

		union
		{
			const std::string strKey;
			const lua_Integer intKey;
		};

		const bool wasKeyInt;

		StackItemRef(lua_State* L, const int idx, const std::string& key)
			:StackItem(L, idx), strKey(key), wasKeyInt(false)
		{}
		StackItemRef(lua_State* L, const int idx, const lua_Integer& key)
			:StackItem(L, idx), intKey(key), wasKeyInt(true)
		{}

		//normal
		StackItem& set(const slua::Pushable auto& newVal)
		{
			setStack(newVal);

			//-3, is also the position the table will be at
			if (parentIdx == -3)
				lua_pushglobaltable(L);

			if (wasKeyInt)
			{
				lua_pushvalue(L, idx);//make a copy
				lua_seti(L, parentIdx, intKey);
			}
			else
			{
				lua_pushlstring(L, strKey.data(), strKey.size());
				lua_pushvalue(L, idx);//make a copy
				lua_settable(L, parentIdx);
			}

			if (parentIdx == -3)
				lua_pop();//remove the global table

			return *this;
		}

		//raw
		StackItem& operator=(const slua::Pushable auto& newVal)
		{
			StackItem::operator=(newVal);

			//-3, is also the position the table will be at
			if (parentIdx == -3)
				lua_pushglobaltable(L);

			if (wasKeyInt)
			{
				lua_pushvalue(L, idx);//make a copy
				lua_rawseti(L, parentIdx, intKey);
			}
			else
			{
				lua_pushlstring(L, strKey.data(), strKey.size());
				lua_pushvalue(L, idx);//make a copy
				lua_rawset(L, parentIdx);
			}


			if (parentIdx == -3)
				lua_pop();//remove the global table

			return *this;
		}

		//handle string inside union
		~StackItemRef()
		{
			if (!wasKeyInt)
				~strKey();
			else
				~intKey();
		}
	};



	StackItemRef makeStackItemRef(lua_State* _L, const lua_Integer key) {
		return StackItemRef(L, lua_gettop(L), key);
	}
	StackItemRef makeStackItemRef(lua_State* _L, const std::string& key) {
		return StackItemRef(L, lua_gettop(L), key);
	}
}
