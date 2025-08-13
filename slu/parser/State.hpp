/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>
#include <optional>
#include <memory>
#include <variant>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/ext/CppMatch.hpp>
#include <slu/ext/ExtendVariant.hpp>
#include <slu/lang/BasicState.hpp>
#include "Enums.hpp"
#include "SmallEnumList.hpp"
#include "Input.hpp"
#include "StateDecls.hpp"
#include "ResolvedType.hpp"

namespace slu::parse
{
	namespace FieldType { using NONE = std::monostate; }

	template<bool isSlu>
	using FieldV = std::variant<
		FieldType::NONE,// Here, so variant has a default value (DO NOT USE)

		FieldType::Expr2ExprV<isSlu>, // "'[' exp ']' = exp"
		FieldType::Name2ExprV<isSlu>, // "Name = exp"
		FieldType::ExprV<isSlu>       // "exp"
	>;
	Slu_DEF_CFG(Field);

	// ‘{’ [fieldlist] ‘}’
	template<bool isSlu>
	using TableV = std::vector<FieldV<isSlu>>;
	Slu_DEF_CFG(Table);

	template<bool isSlu>
	using StatListV = std::vector<StatementV<isSlu>>;
	Slu_DEF_CFG(StatList);


	struct LocalId
	{
		size_t v = SIZE_MAX;
		constexpr bool empty() const { return v == SIZE_MAX; }
		constexpr static LocalId newEmpty() { return LocalId(); }
	};
	template<bool isSlu, bool isLocal>
	using LocalOrNameV = Sel<isLocal, MpItmIdV<isSlu>, LocalId>;
	Slu_DEF_CFG2(LocalOrName, isLocal);
	template<bool isSlu>
	struct LocalsV
	{
		std::vector<MpItmIdV<isSlu>> names;
		std::vector<parse::ResolvedType> types;//Empty until type checking.
	};
	Slu_DEF_CFG(Locals);

	template<bool isSlu>
	using DynLocalOrNameV = Sel<isSlu, MpItmIdV<false>, std::variant<MpItmIdV<true>, LocalId>>;

	enum class RetType : uint8_t
	{
		NONE,
		RETURN,
		BREAK
	};

	template<bool isSlu>
	struct BlockV
	{
		StatListV<isSlu> statList;
		ExprListV<isSlu> retExprs;// May contain 0 elements

		lang::ModPathId mp;

		Position start;
		Position end;

		RetType retTy = RetType::NONE;

		bool empty() const {
			return retTy == RetType::NONE && statList.empty();
		}

		BlockV() = default;
		BlockV(const BlockV&) = delete;
		BlockV(BlockV&&) = default;
		BlockV& operator=(BlockV&&) = default;
	};
	Slu_DEF_CFG(Block);

	namespace SoeType
	{
		template<bool isSlu>
		using BlockV = BlockV<isSlu>;
		Slu_DEF_CFG(Block);

		template<bool isSlu>
		using ExprV = BoxExprV<isSlu>;
		Slu_DEF_CFG(Expr);
	}
	template<bool isSlu>
	using SoeV = std::variant<
		SoeType::BlockV<isSlu>,
		SoeType::ExprV<isSlu>
	>;
	Slu_DEF_CFG(Soe);

	template<bool isSlu> using SoeOrBlockV = Sel<isSlu, BlockV<isSlu>, SoeV<isSlu>>;
	Slu_DEF_CFG(SoeOrBlock);

	template<bool isSlu> using SoeBoxOrBlockV = Sel<isSlu, BlockV<isSlu>, std::unique_ptr<SoeV<isSlu>>>;
	Slu_DEF_CFG(SoeBoxOrBlock);

	namespace ArgsType
	{
		using parse::ExprListV;
		using parse::ExprList;

		using parse::TableV;
		using parse::Table;

		struct String { std::string v; Position end; };// "LiteralString"
	};
	template<bool isSlu>
	using ArgsV = std::variant<
		ArgsType::ExprListV<isSlu>,
		ArgsType::TableV<isSlu>,
		ArgsType::String
	>;
	Slu_DEF_CFG(Args);


	template<bool isSlu,bool boxed>
	struct ExprUserExprV {
		MayBox<boxed,ExprV<isSlu>> v;
	};
	template<bool boxed>
	struct ExprUserExprV<true, boxed>
	{
		MayBox<boxed, ExprV<true>> v;
		parse::ResolvedType ty;
	};
	Slu_DEF_CFG2(ExprUserExpr,boxed);

	template<bool isSlu, bool boxed> // exp args
	struct CallV : ExprUserExprV<isSlu, boxed>
	{
		ArgsV<isSlu> args;
	};
	Slu_DEF_CFG2(Call, boxed);
	template<bool isSlu, bool boxed> //Lua: exp ":" Name args //Slu: exp "." Name args
	struct SelfCallV : ExprUserExprV<isSlu, boxed>
	{
		ArgsV<isSlu> args;
		MpItmIdV<isSlu> method;//may be unresolved, or itm=trait-fn, or itm=fn
	};
	Slu_DEF_CFG2(SelfCall, boxed);

	namespace ExprType
	{
		struct MpRoot {};		// ":>"
		using Local = LocalId;
		template<bool isSlu>
		using GlobalV = MpItmIdV<isSlu>;
		Slu_DEF_CFG(Global);

		template<bool isSlu> // "(" exp ")"
		using ParensV = BoxExprV<isSlu>;
		Slu_DEF_CFG(Parens);

		struct Deref : ExprUserExprV<true, true> {};// exp ".*"

		template<bool isSlu> // exp "[" exp "]"
		struct IndexV : ExprUserExprV<isSlu, true> {
			MayBox<true,ExprV<isSlu>> idx;
		};
		Slu_DEF_CFG(Index);

		template<bool isSlu> // exp "." Name
		struct FieldV : ExprUserExprV<isSlu, true> {
			PoolString field;
		};
		Slu_DEF_CFG(Field);

		template<bool isSlu>
		using CallV = parse::CallV<isSlu, true>;
		Slu_DEF_CFG(Call);
		template<bool isSlu>
		using SelfCallV = parse::SelfCallV<isSlu, true>;
		Slu_DEF_CFG(SelfCall);
	}

	struct TupleName
	{
		uint64_t lo = 0; uint64_t hi = 0;
		constexpr TupleName() = default;
		constexpr TupleName(ExprType::I64 v)
			:lo(v) {}
		constexpr TupleName(ExprType::U64 v)
			: lo(v) {}
		TupleName(ExprType::M128 v) { throw std::runtime_error("Cant index into tuple with negative index."); }
		constexpr TupleName(ExprType::P128 v)
			: lo(v.lo), hi(v.hi) {}
	};

	template<class T>
	concept Any128BitInt =
		std::same_as<T, ExprType::P128>
		|| std::same_as<T, ExprType::M128>;
	template<class T>
	concept Any64BitInt =
		std::same_as<T, ExprType::U64>
		|| std::same_as<T, ExprType::I64>;

	//Slu
	using Lifetime = std::vector<MpItmIdV<true>>;
	struct UnOpItem
	{
		Lifetime life;
		UnOpType type;
	};

	struct TraitExpr
	{
		std::vector<ExprV<true>> traitCombo;
		Position place;
	};
	using TypePrefix = std::vector<UnOpItem>;


	//Common

	//NOTE: has overload later!!!
	template<bool isSlu>
	struct ParameterV
	{
		MpItmIdV<isSlu> name;
	};
	Slu_DEF_CFG(Parameter);

	template<bool isSlu>
	using ParamListV = std::vector<ParameterV<isSlu>>;
	Slu_DEF_CFG(ParamList);

	template<bool isSlu>
	struct FunctionInfoV
	{
		ParamListV<isSlu> params;
		bool hasVarArgParam = false;// do params end with '...'
	};
	template<>
	struct FunctionInfoV<true>
	{
		std::string abi;
		LocalsV<true> local2Mp;
		ParamListV<true> params;
		std::optional<BoxExprV<true>> retType;
		bool hasVarArgParam = false;// do params end with '...'
		OptSafety safety = OptSafety::DEFAULT;
	};
	Slu_DEF_CFG(FunctionInfo);

	template<bool isSlu>
	struct FunctionV : FunctionInfoV<isSlu>
	{
		BlockV<isSlu> block;
	};
	Slu_DEF_CFG(Function);



	template<bool isSlu, bool boxIt>
	struct BaseIfCondV
	{
		std::vector<std::pair<ExprV<isSlu>, SoeOrBlockV<isSlu>>> elseIfs;
		MayBox<boxIt, ExprV<isSlu>> cond;
		MayBox<boxIt, SoeOrBlockV<isSlu>> bl;
		std::optional<MayBox<boxIt, SoeOrBlockV<isSlu>>> elseBlock;
	};
	Slu_DEF_CFG2(BaseIfCond, boxIt);

	namespace ExprType
	{
		using Nil = std::monostate;								// "nil"
		struct False {};										// "false"
		struct True {};											// "true"
		struct VarArgs {};										// "..."

		using parse::FunctionV;
		using parse::Function;
		
		using parse::TableV;
		using parse::Table;

		//unOps is always empty for this type
		template<bool isSlu>
		struct MultiOpV
		{
			BoxExprV<isSlu> first;
			std::vector<std::pair<BinOpType, ExprV<isSlu>>> extra;//size>=1
		};      // "exp binop exp"
		Slu_DEF_CFG(MultiOp);

		template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, true>;
		Slu_DEF_CFG(IfCond);

		using parse::Lifetime;	// " '/' var" {'/' var"}
		using parse::TraitExpr;

		struct PatTypePrefix {};

		struct Inferr {};
		struct Struct
		{
			TableV<true> fields;
		};
		struct Union
		{
			TableV<true> fields;
		};
		struct FnType
		{
			BoxExprV<true> argType;
			BoxExprV<true> retType;
			OptSafety safety = OptSafety::DEFAULT;
		};
		struct Dyn {
			parse::TraitExpr expr;
		};
		struct Impl {
			parse::TraitExpr expr;
		};
		struct Slice {
			BoxExprV<true> v;
		};
		struct Err
		{
			BoxExprV<true> err;
		};
	}

	template<bool isSlu>
	using ExprDataV = std::variant <
		ExprType::Nil,                  // "nil"
		ExprType::False,                // "false"
		ExprType::True,                 // "true"
		ExprType::F64,				// "Numeral" (e.g., a floating-point number)
		ExprType::I64,			// "Numeral"

		ExprType::String,		// "LiteralString"
		ExprType::VarArgs,              // "..." (varargs)
		ExprType::FunctionV<isSlu>,			// "functiondef"
		ExprType::TableV<isSlu>,	// "tableconstructor"

		ExprType::MultiOpV<isSlu>,		// "exp binop exp {binop exp}"  // added {binop exp}, cuz multi-op

		ExprType::Local,
		ExprType::GlobalV<isSlu>,

		ExprType::ParensV<isSlu>,
		ExprType::Deref,

		ExprType::IndexV<isSlu>,
		ExprType::FieldV<isSlu>,
		ExprType::CallV<isSlu>,
		ExprType::SelfCallV<isSlu>,

		// Slu

		ExprType::MpRoot,

		ExprType::IfCondV<isSlu>,

		ExprType::OpenRange,			// ".."

		ExprType::U64,			// "Numeral"
		ExprType::P128,			// "Numeral"
		ExprType::M128,			// "Numeral"

		ExprType::Lifetime,
		ExprType::TraitExpr,

		ExprType::PatTypePrefix,

		// types

		ExprType::Inferr,
		ExprType::Struct,
		ExprType::Union,
		ExprType::Dyn,
		ExprType::Impl,
		ExprType::Slice,
		ExprType::Err,
		ExprType::FnType
	> ;
	Slu_DEF_CFG(ExprData);


	template<bool isSlu>
	struct BaseExprV
	{
		ExprDataV<isSlu> data;
		Position place;
		std::vector<UnOpItem> unOps;//TODO: for lua, use small op list

		BaseExprV() = default;
		BaseExprV(ExprDataV<isSlu>&& data):data(std::move(data)) {}
		BaseExprV(ExprDataV<isSlu>&& data,Position place):data(std::move(data)),place(place) {}

		BaseExprV(const BaseExprV&) = delete;
		BaseExprV(BaseExprV&&) = default;
		BaseExprV& operator=(BaseExprV&&) = default;
	};

	template<bool isSlu>
	struct ExprV : BaseExprV<isSlu>
	{};
	template<>
	struct ExprV<true> : BaseExprV<true>
	{
		SmallEnumList<PostUnOpType> postUnOps;

		bool isBasicStruct() const {
			if(!this->unOps.empty() || !this->postUnOps.empty())
				return false;
			return std::holds_alternative<ExprType::TableV<true>>(this->data);
		}
	};

	//Slu


	// match patterns

	template<bool isSlu>
	using NdPatV = ExprV<isSlu>;
	Slu_DEF_CFG(NdPat);

	namespace DestrSpecType
	{
		template<bool isSlu>
		using SpatV = parse::NdPatV<isSlu>;
		Slu_DEF_CFG(Spat);
		using Prefix = TypePrefix;
	}
	template<bool isSlu>
	using DestrSpecV = std::variant<
		DestrSpecType::SpatV<isSlu>,
		DestrSpecType::Prefix
	>;
	Slu_DEF_CFG(DestrSpec);
	namespace DestrPatType
	{
		template<bool isSlu, bool isLocal>
		using AnyV = LocalOrNameV<isSlu,isLocal>;
		Slu_DEF_CFG2(Any, isLocal);

		template<bool isSlu, bool isLocal> struct FieldsV;
		Slu_DEF_CFG2(Fields, isLocal);
		template<bool isSlu, bool isLocal> struct ListV;
		Slu_DEF_CFG2(List, isLocal);

		template<bool isSlu, bool isLocal>struct NameV;
		Slu_DEF_CFG2(Name, isLocal);
		template<bool isSlu, bool isLocal>struct NameRestrictV;
		Slu_DEF_CFG2(NameRestrict, isLocal);
	}

	namespace PatType
	{
		//x or y or z
		template<bool isSlu>
		using SimpleV = NdPatV<isSlu>;
		Slu_DEF_CFG(Simple);

		template<bool isSlu, bool isLocal>
		using DestrAnyV = DestrPatType::AnyV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrAny, isLocal);

		template<bool isSlu,bool isLocal>
		using DestrFieldsV = DestrPatType::FieldsV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrFields, isLocal);
		template<bool isSlu, bool isLocal>
		using DestrListV = DestrPatType::ListV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrList, isLocal);

		template<bool isSlu, bool isLocal>
		using DestrNameV = DestrPatType::NameV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrName, isLocal);

		template<bool isSlu, bool isLocal>
		using DestrNameRestrictV = DestrPatType::NameRestrictV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrNameRestrict, isLocal);
	}
	template<bool isLocal,typename T>
	concept AnyCompoundDestr =
		std::same_as<std::remove_cv_t<T>, DestrPatType::FieldsV<true, isLocal>>
		|| std::same_as<std::remove_cv_t<T>, DestrPatType::ListV<true, isLocal>>;


	template<bool isSlu,bool isLocal>
	using PatV = std::variant<
		PatType::DestrAnyV<isSlu, isLocal>,

		PatType::SimpleV<isSlu>,

		PatType::DestrFieldsV<isSlu, isLocal>,
		PatType::DestrListV<isSlu, isLocal>,

		PatType::DestrNameV<isSlu, isLocal>,
		PatType::DestrNameRestrictV<isSlu, isLocal>
	>;
	Slu_DEF_CFG2(Pat, isLocal);

	template<bool isSlu, bool isLocal>
	struct DestrFieldV
	{
		PoolString name;// |(...)| thingy
		PatV<isSlu,isLocal> pat;//May be any type of pattern
	};
	Slu_DEF_CFG2(DestrField,isLocal);
	namespace DestrPatType
	{
		template<bool isSlu, bool isLocal>
		struct FieldsV
		{
			DestrSpecV<isSlu> spec;
			bool extraFields : 1 = false;
			std::vector<DestrFieldV<isSlu, isLocal>> items;
			LocalOrNameV<isSlu, isLocal> name;//May be synthetic
		};
		template<bool isSlu, bool isLocal>
		struct ListV
		{
			DestrSpecV<isSlu> spec;
			bool extraFields : 1 = false;
			std::vector<PatV<isSlu, isLocal>> items;
			LocalOrNameV<isSlu,isLocal> name;//May be synthetic
		};

		template<bool isSlu, bool isLocal>
		struct NameV
		{
			LocalOrNameV<isSlu, isLocal> name;
			DestrSpecV<isSlu> spec;
		};
		template<bool isSlu, bool isLocal>
		struct NameRestrictV : NameV<isSlu, isLocal>
		{
			NdPatV<isSlu> restriction;
		};
	}

	template<>
	struct ParameterV<true>
	{
		LocalId name;
		ExprV<true> type;
	};
	template<bool isLocal>
	struct ___PatHack : PatV<true, isLocal> {};

	//Common

	namespace FieldType
	{
		template<bool isSlu>
		struct Expr2ExprV { parse::ExprV<isSlu> idx; parse::ExprV<isSlu> v; };		// "‘[’ exp ‘]’ ‘=’ exp"

		template<bool isSlu>
		struct Name2ExprV { MpItmIdV<isSlu> idx; parse::ExprV<isSlu> v; };	// "Name ‘=’ exp"
	}

	template<bool isSlu>
	struct AttribNameV
	{
		MpItmIdV<isSlu> name;
		std::string attrib;//empty -> no attrib
	};
	Slu_DEF_CFG(AttribName);
	template<bool isSlu>
	using AttribNameListV = std::vector<AttribNameV<isSlu>>;
	Slu_DEF_CFG(AttribNameList);
	template<bool isSlu>
	using NameListV = std::vector<MpItmIdV<isSlu>>;
	Slu_DEF_CFG(NameList);

	namespace UseVariantType
	{
		using EVERYTHING_INSIDE = std::monostate;//use x::*;
		struct IMPORT { MpItmIdV<true> name; };// use x::y; //name is inside this mp, base is the imported path.
		using AS_NAME = MpItmIdV<true>;//use x as y;
		using LIST_OF_STUFF = std::vector<MpItmIdV<true>>;//use x::{self, ...}
	}
	using UseVariant = std::variant<
		UseVariantType::EVERYTHING_INSIDE,
		UseVariantType::AS_NAME,
		UseVariantType::IMPORT,
		UseVariantType::LIST_OF_STUFF
	>;

	struct StructBase
	{
		ParamListV<true> params;
		LocalsV<true> local2Mp;
		TableV<true> type;
		MpItmIdV<true> name;
		ExportData exported = false;
	};
	template<bool isSlu, bool isLocal>
	struct MaybeLocalsV {
		LocalsV<isSlu> local2Mp;
	};
	template<bool isSlu>
	struct MaybeLocalsV<isSlu,true> {};
	template<bool isSlu, bool isLocal>
	struct VarStatBaseV : MaybeLocalsV<isSlu, isLocal>
	{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
		AttribNameListV<isSlu> names;
		ExprListV<isSlu> exprs;
	};
	template<bool isLocal>
	struct VarStatBaseV<true, isLocal> : MaybeLocalsV<true,isLocal>
	{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
		PatV<true, isLocal> names;
		ExprListV<true> exprs;
		ExportData exported = false;
	};

	struct WhereClause
	{
		TraitExpr bound;
		MpItmIdV<true> var;
	};
	using WhereClauses = std::vector<WhereClause>;

	namespace StatementType
	{
		using Semicol = std::monostate;	// ";"

		template<bool isSlu>
		struct AssignV { std::vector<ExprDataV<isSlu>> vars; ExprListV<isSlu> exprs; };// "varlist = explist" //e.size must be > 0
		Slu_DEF_CFG(Assign);

		template<bool isSlu>
		using CallV = parse::CallV<isSlu, false>;
		Slu_DEF_CFG(Call);
		template<bool isSlu>
		using SelfCallV = parse::SelfCallV<isSlu, false>;
		Slu_DEF_CFG(SelfCall);


		template<bool isSlu>
		struct LabelV { MpItmIdV<isSlu> v; };		// "label"
		Slu_DEF_CFG(Label);
		struct Break {};					// "break"
		template<bool isSlu>
		struct GotoV { MpItmIdV<isSlu> v; };			// "goto Name"
		Slu_DEF_CFG(Goto);

		using parse::BlockV;// "do block end"
		using parse::Block;

		template<bool isSlu>
		struct WhileV { ExprV<isSlu> cond; BlockV<isSlu> bl; };		// "while exp do block end"
		Slu_DEF_CFG(While);

		template<bool isSlu>
		struct RepeatUntilV :WhileV<isSlu> {};						// "repeat block until exp"
		Slu_DEF_CFG(RepeatUntil);

		// "if exp then block {elseif exp then block} [else block] end"
		template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, false>;
		Slu_DEF_CFG(IfCond);

		// "for Name = exp , exp [, exp] do block end"
		template<bool isSlu>
		struct ForNumV
		{
			Sel<isSlu, MpItmIdV<isSlu>, PatV<true,true>> varName;
			ExprV<isSlu> start;
			ExprV<isSlu> end;//inclusive
			std::optional<ExprV<isSlu>> step;
			BlockV<isSlu> bl;
		};
		Slu_DEF_CFG(ForNum);

		// "for namelist in explist do block end"
		template<bool isSlu>
		struct ForInV
		{
			Sel<isSlu, NameListV<isSlu>, PatV<true, true>> varNames;
			Sel<isSlu, ExprListV<isSlu>, ExprV<isSlu>> exprs;//size must be > 0
			BlockV<isSlu> bl;
		};
		Slu_DEF_CFG(ForIn);

		template<bool isSlu>
		struct FuncDefBase
		{// "function funcname funcbody"    
			Position place;//Right after func-name
			MpItmIdV<isSlu> name; // name may contain dots, 1 colon if !isSlu
			FunctionV<isSlu> func;
		};
		template<bool isSlu>
		struct FunctionV : FuncDefBase<isSlu> {};
		template<>
		struct FunctionV<true> : FuncDefBase<true>
		{
			ExportData exported = false;
		};
		Slu_DEF_CFG(Function);

		template<bool isSlu>
		struct FnV : FunctionV<isSlu> {};
		Slu_DEF_CFG(Fn);

		template<bool isSlu>
		struct LocalFunctionDefV :FunctionV<isSlu> {};
		Slu_DEF_CFG(LocalFunctionDef);
		// "local function Name funcbody"


		template<bool isSlu>
		struct FunctionDeclV : FunctionInfoV<isSlu>
		{
			Position place;//Right before func-name
			MpItmIdV<isSlu> name;
			ExportData exported = false;
		};
		Slu_DEF_CFG(FunctionDecl);

		template<bool isSlu>
		struct FnDeclV : FunctionDeclV<isSlu> {};
		Slu_DEF_CFG(FnDecl);

		template<bool isSlu>
		using LocalV = VarStatBaseV<isSlu,true>;
		Slu_DEF_CFG(Local);

		// Slu

		template<bool isSlu>
		struct LetV : LocalV<isSlu> {};
		Slu_DEF_CFG(Let);

		template<bool isSlu>
		using ConstV = VarStatBaseV<isSlu, false>;
		Slu_DEF_CFG(Const);

		struct CanonicLocal
		{
			ResolvedType type;
			LocalId name;
			ExprV<true> value;
			ExportData exported = false;
		};
		struct CanonicGlobal
		{
			ResolvedType type;
			LocalsV<true> local2Mp;
			MpItmIdV<true> name;
			ExprV<true> value;
			ExportData exported = false;
		};


		struct Struct : StructBase {};
		struct Union : StructBase {};

		template<bool isSlu>
		struct ExternBlockV
		{
			StatListV<isSlu> stats;
			std::string abi;
			Position abiEnd;
			OptSafety safety = OptSafety::DEFAULT;
		};
		Slu_DEF_CFG(ExternBlock);

		template<bool isSlu>
		struct UnsafeBlockV {
			StatListV<isSlu> stats;
		};	// "unsafe {...}"
		Slu_DEF_CFG(UnsafeBlock);

		struct UnsafeLabel {};
		struct SafeLabel {};

		struct Trait
		{
			WhereClauses clauses;
			MpItmIdV<true> name;
			ParamListV<true> params;
			StatListV<true> itms;
			std::optional<TraitExpr> whereSelf;
			ExportData exported = false;
		};
		struct Impl
		{
			WhereClauses clauses;
			ParamListV<true> params;
			std::optional<TraitExpr> forTrait;
			ExprV<true> type;
			StatListV<true> code;
			ExportData exported: 1 = false;
			bool deferChecking : 1 = false;
			bool isUnsafe	   : 1 = false;
		};

		template<bool isSlu>
		struct DropV
		{
			ExprV<isSlu> expr;
		};
		Slu_DEF_CFG(Drop);


		struct Use
		{
			MpItmIdV<true> base;//the aliased/imported thing, or modpath base
			UseVariant useVariant;
			ExportData exported = false;
		};

		template<bool isSlu>
		struct ModV
		{
			MpItmIdV<isSlu> name;
			ExportData exported = false;
		};
		Slu_DEF_CFG(Mod);

		template<bool isSlu>
		struct ModAsV : ModV<isSlu>
		{
			StatListV<isSlu> code;
		};
		Slu_DEF_CFG(ModAs);

	};

	template<bool isSlu>
	using StatementDataV = std::variant <
		StatementType::Semicol,				// ";"

		StatementType::AssignV<isSlu>,			// "varlist = explist"
		StatementType::LocalV<isSlu>,	// "local attnamelist [= explist]"
		StatementType::LetV<isSlu>,	// "let pat [= explist]"
		StatementType::ConstV<isSlu>,	// "const pat [= explist]"
		StatementType::CanonicLocal,
		StatementType::CanonicGlobal,

		StatementType::CallV<isSlu>,
		StatementType::SelfCallV<isSlu>,

		StatementType::LabelV<isSlu>,			// "label"
		StatementType::Break,					// "break"
		StatementType::GotoV<isSlu>,			// "goto Name"
		StatementType::BlockV<isSlu>,			// "do block end"
		StatementType::WhileV<isSlu>,		// "while exp do block end"
		StatementType::RepeatUntilV<isSlu>,	// "repeat block until exp"

		StatementType::IfCondV<isSlu>,	// "if exp then block {elseif exp then block} [else block] end"

		StatementType::ForNumV<isSlu>,	// "for Name = exp , exp [, exp] do block end"
		StatementType::ForInV<isSlu>,	// "for namelist in explist do block end"

		StatementType::LocalFunctionDefV<isSlu>,	// "local function Name funcbody"
		StatementType::FunctionV<isSlu>,		// "function funcname funcbody"
		StatementType::FnV<isSlu>,					// "fn funcname funcbody"

		StatementType::FunctionDeclV<isSlu>,
		StatementType::FnDeclV<isSlu>,

		StatementType::Struct,
		StatementType::Union,

		StatementType::ExternBlockV<isSlu>,

		StatementType::UnsafeBlockV<isSlu>,
		StatementType::UnsafeLabel,	// ::: unsafe :
		StatementType::SafeLabel,		// ::: safe :

		StatementType::DropV<isSlu>,	// "drop" Name
		StatementType::Trait,
		StatementType::Impl,

		StatementType::Use,				// "use" ...
		StatementType::ModV<isSlu>,		// "mod" Name
		StatementType::ModAsV<isSlu>	// "mod" Name "as" "{" block "}"
	> ;
	Slu_DEF_CFG(StatementData);

	template<bool isSlu>
	struct StatementV
	{
		StatementDataV<isSlu> data;
		Position place;

		StatementV() = default;
		StatementV(StatementDataV<isSlu>&& data) :data(std::move(data)) {}
		StatementV(const StatementV&) = delete;
		StatementV(StatementV&&) = default;
		StatementV& operator=(StatementV&&) = default;
	};


	template<bool isSlu>
	struct ParsedFileV
	{
		StatListV<isSlu> code;
		lang::ModPathId mp;
	};
	template<>
	struct ParsedFileV<false>
	{
		BlockV<false> code;
		LocalsV<false> local2Mp;
	};
	Slu_DEF_CFG(ParsedFile);
}