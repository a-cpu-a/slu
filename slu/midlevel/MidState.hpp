﻿/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include <slu/parser/State.hpp>
#include <slu/lang/BasicState.hpp>

namespace slu::mlvl
{
	using slu::lang::ModPath;
	using slu::lang::ModPathView;
	using slu::lang::ExportData;

	/*
	
	Mid level plans:

	---------0---------
	type checker?
	type inference?
	name resolver
	---------1---------
	referenced name checker?
	---------2---------
	safety checker
	?stability checker?
	---------3---------
	purity checker & inference?
	---------4---------
	borrow checker?
	-------------------
	*/
	struct TypeId : MpItmIdV
	{};
	struct ObjId : MpItmIdV
	{};
	struct LocalTypedObjId
	{
		size_t valId;
	};



	struct BasicData
	{
		parse::Position defPos;
		ExportData exData;
	};




	namespace ObjType
	{
		struct FuncId :MpItmIdV {};
		struct VarId :MpItmIdV {};
		struct TypeId :MpItmIdV {};
		struct TraitId :MpItmIdV {};
		struct ImplId :MpItmIdV {};
		struct UseId :MpItmIdV {};
		struct MacroId :MpItmIdV {};
	}
	using Obj = std::variant<
		ObjType::FuncId,
		ObjType::VarId,
		ObjType::TypeId,
		ObjType::TraitId,
		ObjType::ImplId,
		ObjType::UseId,
		ObjType::MacroId
	>;

	namespace LocalObjType
	{
		struct FuncId :LocalTypedObjId {};
		struct VarId :LocalTypedObjId {};
		struct TypeId :LocalTypedObjId {};
		struct TraitId :LocalTypedObjId {};
		struct ImplId :LocalTypedObjId {};
		struct UseId :LocalTypedObjId {};
		struct MacroId :LocalTypedObjId {};
	}
	using LocalObj = std::variant<
		LocalObjType::FuncId,
		LocalObjType::VarId,
		LocalObjType::TypeId,
		LocalObjType::TraitId,
		LocalObjType::ImplId,
		LocalObjType::UseId,
		LocalObjType::MacroId
	>;

	struct Func : BasicData
	{
		std::string name;
		parse::FunctionV<true> data;//todo: new struct for: flattened non-block syntax blocks, ref by "ptr"
	};
	struct Var : BasicData
	{
		// TODO: destructuring
		parse::ExpListV<true> vals;//TODO: same as func
	};
	struct Type : BasicData
	{
		std::string name;
		parse::ErrType val;
	};
	struct Trait : BasicData
	{
		std::string name;
		//TODO
	};
	struct Impl : BasicData
	{
		//TODO
	};
	struct Use : BasicData
	{
		//TODO: make a local copy of use variant, with a better data format, with support for id ptrs
		ModPathId base;
		parse::UseVariant value;
	};
	struct Macro : BasicData
	{
		std::string name;
		//TODO
	};

	struct MpData : BasicData
	{
		ModPath path;

		std::vector<Func> funcs;
		std::vector<Var> vars;
		std::vector<Type> types;
		std::vector<Trait> traits;
		std::vector<Impl> impl;
		std::vector<Use> uses;
		std::vector<Macro> macros;

		std::vector<LocalObj> objs;//MpData::_[...]

		std::vector<ModPathId> subModules;
		bool isInline : 1 = true;
	};
	struct CrateData
	{
		std::string name;
		ModPathId root;
	};
	struct MidState
	{
		std::vector<MpData> modPaths;
		std::vector<CrateData> crates;
	};
}