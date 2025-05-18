/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <variant>
#include <slu/Include.hpp>

#include <slu/Utils.hpp>
#include <slu/types/Converter.hpp>
#include <slu/types/ReadWrite.hpp>
#include <slu/types/complex/String.hpp>
#include <slu/types/basic/Integer.hpp>
#include <slu/ext/CppMatch.hpp>


namespace slua { struct StackItemRef; }//Forward decl

//Turns the item on the top of the stack into a StackItemRef
slua::StackItemRef Slua_makeStackItemRef(lua_State* _L, const lua_Integer key, const int parentIdx);
//Turns the item on the top of the stack into a StackItemRef
slua::StackItemRef Slua_makeStackItemRef(lua_State* _L, const std::string& key, const int parentIdx);

namespace slua
{
	// Note: dont mix scopes
	struct StackItem
	{
		lua_State* L=nullptr;

		const int idx = 0;

		constexpr StackItem() = default;
		constexpr StackItem(lua_State* _L, const int _idx) :L(_L), idx(_idx) {}

		//normal
		const StackItem& set(const slua::Pushable auto& newVal, const lua_Integer key)const
		{
			slua::push(newVal);
			lua_seti(L, idx, key);
			return *this;
		}
		//normal
		const StackItem& set(const slua::Pushable auto& newVal, const std::string& key)const
		{
			slua::push(key);
			slua::push(newVal);
			lua_settable(L, idx);
			return *this;
		}

		//normal, but will raw-set by default
		slua::StackItemRef get(const lua_Integer key)const;
		//normal, but will raw-set by default
		slua::StackItemRef get(const std::string& key)const;

		//raw
		slua::StackItemRef operator[](const lua_Integer key)const;
		//raw
		slua::StackItemRef operator[](const std::string& key)const;

		//normal
		std::string toString()const {
			size_t len;
			return std::string(lua_tolstring(L, idx, &len), len);
		}

		lua_Unsigned rawLen() const {
			return lua_rawlen(L, idx);
		}

		//normal
		bool equals(const StackItem& o)const {

			return L == o.L && lua_compare(L, idx, o.idx, LUA_OPEQ);
		}
		//raw
		bool operator==(const StackItem& o) const {
			return L == o.L && lua_rawequal(L, idx, o.idx);
		}

		//normal
		bool lessThan(const StackItem& o)const {
			return L == o.L && lua_compare(L, idx, o.idx, LUA_OPLT) != 0;
		}
		//"raw"
		bool operator<(const StackItem& o) const {
			if (!lua_isnumber(L, idx) || !lua_isnumber(o.L, idx))
				return false;//cant compare other types

			return lua_tonumber(L, idx) < lua_tonumber(o.L, o.idx);
		}
		//normal
		bool lessThanEqual(const StackItem& o) const {
			return L == o.L && lua_compare(L, idx, o.idx, LUA_OPLE) != 0;
		}
		//"raw"
		bool operator<=(const StackItem& o) const {
			if (!lua_isnumber(L, idx) || !lua_isnumber(o.L, idx))
				return false;//cant compare other types

			return lua_tonumber(L, idx) <= lua_tonumber(o.L, o.idx);
		}

		const StackItem& setStack(const slua::Pushable auto& newVal)const
		{
			slua::push(L, newVal);
			lua_replace(L, idx);// replace this item with the value just pushed

			return *this;
		}

		//raw
		const StackItem& operator=(const slua::Pushable auto& newVal)const {
			setStack(newVal);
			return *this;
		}

		~StackItem() {
			lua_pop(L, idx);
		}
	};

	struct StackItemRef : StackItem
	{
		const std::variant<std::string, lua_Integer> key;

		const int parentIdx;// -3 -> _G, LUA_REGISTRYINDEX -> registry

		constexpr StackItemRef(lua_State* L, const int idx, const std::string& key, const int parentIdx)
			:StackItem(L, idx), key(key), parentIdx(parentIdx)
		{}
		constexpr StackItemRef(lua_State* L, const int idx, const lua_Integer& key, const int parentIdx)
			:StackItem(L, idx), key(key), parentIdx(parentIdx)
		{}

		//normal
		StackItem& set(const slua::Pushable auto& newVal)
		{
			setStack(newVal);

			//-3, is also the position the table will be at
			if (parentIdx == -3)
				lua_pushglobaltable(L);

			ezmatch(key)(
				varcase(const lua_Integer) {
					lua_pushvalue(L, idx);//make a copy
					lua_seti(L, parentIdx, var);
				},
				varcase(const std::string) {
					lua_pushlstring(L, var.data(), var.size());
					lua_pushvalue(L, idx);//make a copy
					lua_settable(L, parentIdx);
				}
			);

			if (parentIdx == -3)
				lua_pop(L,1);//remove the global table

			return *this;
		}

		//raw
		StackItem& operator=(const slua::Pushable auto& newVal)
		{
			StackItem::operator=(newVal);

			//-3, is also the position the table will be at
			if (parentIdx == -3)
				lua_pushglobaltable(L);

			ezmatch(key)(
				varcase(const lua_Integer){
					lua_pushvalue(L, idx);//make a copy
					lua_rawseti(L, parentIdx, var);
				},
				varcase(const std::string){
					lua_pushlstring(L, var.data(), var.size());
					lua_pushvalue(L, idx);//make a copy
					lua_rawset(L, parentIdx);
				}
			);


			if (parentIdx == -3)
				lua_pop(L,1);//remove the global table

			return *this;
		}
	};


	//normal, but will raw-set by default
	inline slua::StackItemRef slua::StackItem::get(const lua_Integer key) const
	{
		lua_geti(L, idx, key);
		return Slua_makeStackItemRef(L, key, idx);
	}
	//normal, but will raw-set by default
	inline slua::StackItemRef slua::StackItem::get(const std::string& key)const
	{
		lua_pushlstring(L, key.data(), key.size());
		lua_gettable(L, idx);
		return Slua_makeStackItemRef(L, key, idx);
	}

	//raw
	slua::StackItemRef slua::StackItem::operator[](const lua_Integer key)const
	{
		lua_rawgeti(L, idx, key);
		return Slua_makeStackItemRef(L, key, idx);
	}
	//raw
	slua::StackItemRef slua::StackItem::operator[](const std::string& key)const
	{
		lua_pushlstring(L, key.data(), key.size());
		lua_rawget(L, idx);
		return Slua_makeStackItemRef(L, key, idx);
	}

}
slua::StackItemRef Slua_makeStackItemRef(lua_State* L, const lua_Integer key, const int parentIdx) {
	return slua::StackItemRef(L, lua_gettop(L), key, parentIdx);
}
slua::StackItemRef Slua_makeStackItemRef(lua_State* L, const std::string& key, const int parentIdx) {
	return slua::StackItemRef(L, lua_gettop(L), key, parentIdx);
}
