/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include <slua/parser/State.hpp>
#include <slua/lang/BasicState.hpp>

namespace slua::mlvl
{
	using slua::lang::ModPath;
	using slua::lang::ExportData;

	//TODO: index by id "ptr", not strings

	namespace ObjType
	{
		struct Func;
		struct Var;
		struct Type;
		struct Trait;
		struct Impl;
		struct Use;
		struct Macro;
		struct Module;
	}
	using Obj = std::variant<
		ObjType::Func,
		ObjType::Var,
		ObjType::Type,
		ObjType::Trait,
		ObjType::Impl,
		ObjType::Use,
		ObjType::Macro,
		ObjType::Module
	>;
	namespace ObjType
	{
		struct Base
		{
			sluaParse::Position defPos;
			ExportData exData;
		};
		struct Func : Base
		{
			std::string name;
			sluaParse::FunctionV<true> data;//todo: new struct for: flattened non-block syntax blocks, ref by "ptr"
		};
		struct Var : Base
		{
			// TODO: destructuring
			sluaParse::ExpListV<true> vals;//TODO: same as func
		};
		struct Type : Base
		{
			std::string name;
			sluaParse::ErrType val;
		};
		struct Trait : Base
		{
			//TODO
		};
		struct Impl : Base
		{
			//TODO
		};
		struct Use : Base
		{
			ModPath base;//the aliased/imported thing, or modpath base
			sluaParse::UseVariant value;
		};
		struct Macro : Base
		{
			//TODO
		};
		struct Module : Base
		{
			std::unordered_map<std::string, Obj> objs;
			bool isInline:1 = true;
		};
	}

	struct CrateData
	{
		std::unordered_map<std::string, ObjType::Module> modules;
	};
	struct MidState
	{
		std::unordered_map<std::string, CrateData> crates;
	};
}