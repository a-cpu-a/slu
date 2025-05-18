/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <string_view>
#include <slua/Include.hpp>

#include <slua/Utils.hpp>
#include <slua/types/Converter.hpp>

namespace slua
{
	struct String
	{
		std::string val;

		constexpr String() = default;
		constexpr String(const std::string& value) :val(value) {}

		static int push(lua_State* L, const String& data)
		{
			lua_pushlstring(L, data.val.data(), data.val.size());
			return 1;
		}
		static String read(lua_State* L, const int idx) {
			return slua::readString(L, idx);
		}
		static bool check(lua_State* L, const int idx) {
			return lua_isstring(L, idx);
		}
		static constexpr const char* getName() { return LC_string; }
	};


	//String view, only for writing!


	struct StringView
	{
		std::string_view val;

		StringView() {}
		StringView(const std::string_view& value) :val(value) {}

		static int push(lua_State* L, const StringView& data)
		{
			lua_pushlstring(L, data.val.data(), data.val.size());
			return 1;
		}

		static constexpr const char* getName() { return LUACC_STRING_DOUBLE "string-view" LUACC_DEFAULT; }
	};


	//Unsafe string, for VERY specific uses, DO NOT USE, unless u need it


	struct DangerString
	{
		std::string_view val;

		size_t size() const { return val.size(); }
		uint8_t operator[](const size_t idx)const { return val[idx]; }

		DangerString() {}
		DangerString(const std::string_view& value) :val(value) {}

		static int push(lua_State* L, const DangerString& data)
		{
			lua_pushlstring(L, data.val.data(), data.val.size());
			return 1;
		}
		static DangerString read(lua_State* L, const int idx) {
			size_t len;
			const char* ptr = lua_tolstring(L, idx, &len);
			return std::string_view(ptr, len);
		}
		static bool check(lua_State* L, const int idx) {
			return lua_isstring(L, idx);
		}
		static constexpr const char* getName() { return LUACC_STRING_DOUBLE "d-string" LUACC_DEFAULT; }
	};


}
// Map basic types to slua::String(View), to allow easy pushing, reading, and checking
Slua_mapType(std::string, slua::String);
Slua_mapType(std::string_view, slua::StringView);