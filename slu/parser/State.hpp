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

		FieldType::Expr2Expr, // "'[' exp ']' = exp"
		FieldType::Name2Expr, // "Name = exp"
		FieldType::Expr       // "exp"
	>;
	Slu_DEF_CFG(Field);


	// ‘{’ [fieldlist] ‘}’
	template<bool isSlu>
	using TableV = std::vector<FieldV<isSlu>>;
	Slu_DEF_CFG(Table);

	template<bool isSlu>
	using StatListV = std::vector<Stat>;
	Slu_DEF_CFG(StatList);


	struct LocalId
	{
		size_t v = SIZE_MAX;
		constexpr bool empty() const { return v == SIZE_MAX; }
		constexpr static LocalId newEmpty() { return LocalId(); }
	};
	template<bool isSlu, bool isLocal>
	using LocalOrNameV = Sel<isLocal, MpItmId, LocalId>;
	Slu_DEF_CFG2(LocalOrName, isLocal);
	template<bool isSlu>
	struct LocalsV
	{
		std::vector<MpItmId> names;
		std::vector<parse::ResolvedType> types;//Empty until type checking.
	};
	Slu_DEF_CFG(Locals);

	template<bool isSlu>
	using DynLocalOrNameV = std::variant<MpItmId, LocalId>;

	enum class RetType : uint8_t
	{
		NONE,
		RETURN,
		BREAK,
		CONTINUE
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

		using Expr = BoxExpr;
	}
	template<bool isSlu>
	using SoeV = std::variant<
		SoeType::BlockV<isSlu>,
		SoeType::Expr
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
	using Args = std::variant<
		ArgsType::ExprListV<true>,
		ArgsType::TableV<true>,
		ArgsType::String
	>;


	template<bool boxed>
	struct ExprUserExpr {
		MayBox<boxed, Expr> v;
		parse::ResolvedType ty;
	};

	template<bool boxed> // exp args
	struct Call : ExprUserExpr<boxed> {
		Args args;
	};
	template<bool boxed> // exp "." Name args
	struct SelfCall : ExprUserExpr<boxed>
	{
		Args args;
		MpItmId method;//may be unresolved, or itm=trait-fn, or itm=fn
	};

	namespace ExprType
	{
		struct MpRoot {};		// ":>"
		using Local = LocalId;
		template<bool isSlu>
		using GlobalV = MpItmId;
		Slu_DEF_CFG(Global);

		template<bool isSlu> // "(" exp ")"
		using ParensV = BoxExpr;
		Slu_DEF_CFG(Parens);

		struct Deref : ExprUserExpr<true> {};// exp ".*"

		struct Index : ExprUserExpr<true> {// exp "[" exp "]"
			MayBox<true,Expr> idx;
		};

		template<bool isSlu> // exp "." Name
		struct FieldV : ExprUserExpr<true> {
			PoolString field;
		};
		Slu_DEF_CFG(Field);

		using Call = parse::Call<true>;
		using SelfCall = parse::SelfCall<true>;
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
	using Lifetime = std::vector<MpItmId>;
	struct UnOpItem
	{
		Lifetime life;
		UnOpType type;
	};

	struct TraitExpr
	{
		std::vector<Expr> traitCombo;
		Position place;
	};
	using TypePrefix = std::vector<UnOpItem>;


	//Common

	template<bool isLocal>
	struct Parameter
	{
		LocalOrNameV<true,isLocal> name;
		Expr type;
	};

	template<bool isLocal>
	using ParamList = std::vector<Parameter<isLocal>>;

	struct FunctionInfo
	{
		std::string abi;
		LocalsV<true> local2Mp;
		ParamList<true> params;
		std::optional<BoxExpr> retType;
		bool hasVarArgParam = false;// do params end with '...'
		OptSafety safety = OptSafety::DEFAULT;
	};
	struct Function : FunctionInfo
	{
		BlockV<true> block;
	};



	template<bool isSlu, bool boxIt>
	struct BaseIfCondV
	{
		std::vector<std::pair<Expr, SoeOrBlockV<isSlu>>> elseIfs;
		MayBox<boxIt, Expr> cond;
		MayBox<boxIt, SoeOrBlockV<isSlu>> bl;
		std::optional<MayBox<boxIt, SoeOrBlockV<isSlu>>> elseBlock;
		parse::ResolvedType ty;
	};
	Slu_DEF_CFG2(BaseIfCond, boxIt);

	namespace ExprType
	{
		using Nil = std::monostate;								// "nil"
		struct False {};										// "false"
		struct True {};											// "true"
		struct VarArgs {};										// "..."

		using parse::Function;
		
		using parse::TableV;
		using parse::Table;

		//unOps is always empty for this type
		template<bool isSlu>
		struct MultiOpV
		{
			BoxExpr first;
			std::vector<std::pair<BinOpType, Expr>> extra;//size>=1
		};      // "exp binop exp"
		Slu_DEF_CFG(MultiOp);

		template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, true>;
		Slu_DEF_CFG(IfCond);

		using parse::Lifetime;	// " '/' var" {'/' var"}
		using parse::TraitExpr;

		struct PatTypePrefix {};

		struct Infer {};
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
			BoxExpr argType;
			BoxExpr retType;
			OptSafety safety = OptSafety::DEFAULT;
		};
		struct Dyn {
			parse::TraitExpr expr;
		};
		struct Impl {
			parse::TraitExpr expr;
		};
		struct Slice {
			BoxExpr v;
		};
		struct Err
		{
			BoxExpr err;
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
		ExprType::Function,			// "functiondef"
		ExprType::TableV<isSlu>,	// "tableconstructor"

		ExprType::MultiOpV<isSlu>,		// "exp binop exp {binop exp}"  // added {binop exp}, cuz multi-op

		ExprType::Local,
		ExprType::GlobalV<isSlu>,

		ExprType::ParensV<isSlu>,
		ExprType::Deref,

		ExprType::Index,
		ExprType::FieldV<isSlu>,
		ExprType::Call,
		ExprType::SelfCall,

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

		ExprType::Infer,
		ExprType::Struct,
		ExprType::Union,
		ExprType::Dyn,
		ExprType::Impl,
		ExprType::Slice,
		ExprType::Err,
		ExprType::FnType
	> ;
	Slu_DEF_CFG(ExprData);



	struct Expr
	{
		ExprDataV<true> data;
		Position place;
		std::vector<UnOpItem> unOps;//TODO: for lua, use small op list

		SmallEnumList<PostUnOpType> postUnOps;

		bool isBasicStruct() const {
			if (!this->unOps.empty() || !this->postUnOps.empty())
				return false;
			return std::holds_alternative<ExprType::TableV<true>>(this->data);
		}

		Expr() = default;
		Expr(ExprDataV<true>&& data) 
			:data(std::move(data)) {}
		Expr(ExprDataV<true>&& data, Position place) 
			:data(std::move(data)), place(place) {}
		Expr(ExprDataV<true>&& data, Position place, std::vector<UnOpItem>&& unOps) 
			:data(std::move(data)), place(place), unOps(std::move(unOps)) {}

		Expr(const Expr&) = delete;
		Expr(Expr&&) = default;
		Expr& operator=(Expr&&) = default;
	};

	//Slu


	// match patterns

	using NdPat = Expr;

	namespace DestrSpecType
	{
		using Spat = parse::NdPat;
		using Prefix = TypePrefix;
	}
	using DestrSpec = std::variant<
		DestrSpecType::Spat,
		DestrSpecType::Prefix
	>;
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
		using SimpleV = NdPat;
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
			DestrSpec spec;
			bool extraFields : 1 = false;
			std::vector<DestrFieldV<isSlu, isLocal>> items;
			LocalOrNameV<isSlu, isLocal> name;//May be synthetic
		};
		template<bool isSlu, bool isLocal>
		struct ListV
		{
			DestrSpec spec;
			bool extraFields : 1 = false;
			std::vector<PatV<isSlu, isLocal>> items;
			LocalOrNameV<isSlu,isLocal> name;//May be synthetic
		};

		template<bool isSlu, bool isLocal>
		struct NameV
		{
			LocalOrNameV<isSlu, isLocal> name;
			DestrSpec spec;
		};
		template<bool isSlu, bool isLocal>
		struct NameRestrictV : NameV<isSlu, isLocal>
		{
			NdPat restriction;
		};
	}

	template<bool isLocal>
	struct ___PatHack : PatV<true, isLocal> {};

	//Common

	namespace FieldType
	{
		struct Expr2Expr { parse::Expr idx; parse::Expr v; };		// "‘[’ exp ‘]’ ‘=’ exp"
		struct Name2Expr { MpItmId idx; parse::Expr v; };	// "Name ‘=’ exp"
	}

	template<bool isSlu>
	struct AttribNameV
	{
		MpItmId name;
		std::string attrib;//empty -> no attrib
	};
	Slu_DEF_CFG(AttribName);
	template<bool isSlu>
	using AttribNameListV = std::vector<AttribNameV<isSlu>>;
	Slu_DEF_CFG(AttribNameList);
	template<bool isSlu>
	using NameListV = std::vector<MpItmId>;
	Slu_DEF_CFG(NameList);

	namespace UseVariantType
	{
		using EVERYTHING_INSIDE = std::monostate;//use x::*;
		struct IMPORT { MpItmId name; };// use x::y; //name is inside this mp, base is the imported path.
		using AS_NAME = MpItmId;//use x as y;
		using LIST_OF_STUFF = std::vector<MpItmId>;//use x::{self, ...}
	}
	using UseVariant = std::variant<
		UseVariantType::EVERYTHING_INSIDE,
		UseVariantType::AS_NAME,
		UseVariantType::IMPORT,
		UseVariantType::LIST_OF_STUFF
	>;

	struct StructBase
	{
		ParamList<true> params;
		LocalsV<true> local2Mp;
		TableV<true> type;
		MpItmId name;
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
		PatV<true, isLocal> names;
		ExprListV<true> exprs;
		ExportData exported = false;
	};

	struct WhereClause
	{
		TraitExpr bound;
		MpItmId var;
	};
	using WhereClauses = std::vector<WhereClause>;

	namespace StatType
	{
		using Semicol = std::monostate;	// ";"

		template<bool isSlu>
		struct AssignV { std::vector<ExprDataV<isSlu>> vars; ExprListV<isSlu> exprs; };// "varlist = explist" //e.size must be > 0
		Slu_DEF_CFG(Assign);

		using Call = parse::Call<false>;
		using SelfCall = parse::SelfCall<false>;


		template<bool isSlu>
		struct LabelV { MpItmId v; };		// "label"
		Slu_DEF_CFG(Label);
		template<bool isSlu>
		struct GotoV { MpItmId v; };			// "goto Name"
		Slu_DEF_CFG(Goto);

		using parse::BlockV;// "do block end"
		using parse::Block;

		template<bool isSlu>
		struct WhileV { Expr cond; BlockV<isSlu> bl; };		// "while exp do block end"
		Slu_DEF_CFG(While);

		template<bool isSlu>
		struct RepeatUntilV :WhileV<isSlu> {};						// "repeat block until exp"
		Slu_DEF_CFG(RepeatUntil);

		// "if exp then block {elseif exp then block} [else block] end"
		template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, false>;
		Slu_DEF_CFG(IfCond);

		// "for namelist in explist do block end"
		template<bool isSlu>
		struct ForInV
		{
			Sel<isSlu, NameListV<isSlu>, PatV<true, true>> varNames;
			Sel<isSlu, ExprListV<isSlu>, Expr> exprs;//size must be > 0
			BlockV<isSlu> bl;
		};
		Slu_DEF_CFG(ForIn);

		template<bool isSlu>
		struct FuncDefBase
		{// "function funcname funcbody"    
			Position place;//Right after func-name
			MpItmId name; // name may contain dots, 1 colon if !isSlu
			Function func;
		};
		struct Function : FuncDefBase<true> {
			ExportData exported = false;
		};

		struct Fn : Function {};

		template<bool isSlu>
		struct FunctionDeclV : FunctionInfo
		{
			Position place;//Right before func-name
			MpItmId name;
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
			Expr value;
			ExportData exported = false;
		};
		struct CanonicGlobal
		{
			ResolvedType type;
			LocalsV<true> local2Mp;
			MpItmId name;
			Expr value;
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
			MpItmId name;
			ParamList<false> params;
			StatListV<true> itms;
			std::optional<TraitExpr> whereSelf;
			ExportData exported = false;
		};
		struct Impl
		{
			WhereClauses clauses;
			ParamList<false> params;
			std::optional<TraitExpr> forTrait;
			Expr type;
			StatListV<true> code;
			ExportData exported: 1 = false;
			bool deferChecking : 1 = false;
			bool isUnsafe	   : 1 = false;
		};

		template<bool isSlu>
		struct DropV
		{
			Expr expr;
		};
		Slu_DEF_CFG(Drop);


		struct Use
		{
			MpItmId base;//the aliased/imported thing, or modpath base
			UseVariant useVariant;
			ExportData exported = false;
		};

		template<bool isSlu>
		struct ModV
		{
			MpItmId name;
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
	using StatDataV = std::variant <
		StatType::Semicol,				// ";"

		StatType::AssignV<isSlu>,			// "varlist = explist"
		StatType::LocalV<isSlu>,	// "local attnamelist [= explist]"
		StatType::LetV<isSlu>,	// "let pat [= explist]"
		StatType::ConstV<isSlu>,	// "const pat [= explist]"
		StatType::CanonicLocal,
		StatType::CanonicGlobal,

		StatType::Call,
		StatType::SelfCall,

		StatType::LabelV<isSlu>,			// "label"
		StatType::GotoV<isSlu>,			// "goto Name"
		StatType::BlockV<isSlu>,			// "do block end"
		StatType::WhileV<isSlu>,		// "while exp do block end"
		StatType::RepeatUntilV<isSlu>,	// "repeat block until exp"

		StatType::IfCondV<isSlu>,	// "if exp then block {elseif exp then block} [else block] end"

		StatType::ForInV<isSlu>,	// "for namelist in explist do block end"

		StatType::Function,		// "function funcname funcbody"
		StatType::Fn,					// "fn funcname funcbody"

		StatType::FunctionDeclV<isSlu>,
		StatType::FnDeclV<isSlu>,

		StatType::Struct,
		StatType::Union,

		StatType::ExternBlockV<isSlu>,

		StatType::UnsafeBlockV<isSlu>,
		StatType::UnsafeLabel,	// ::: unsafe :
		StatType::SafeLabel,		// ::: safe :

		StatType::DropV<isSlu>,	// "drop" Name
		StatType::Trait,
		StatType::Impl,

		StatType::Use,				// "use" ...
		StatType::ModV<isSlu>,		// "mod" Name
		StatType::ModAsV<isSlu>	// "mod" Name "as" "{" block "}"
	> ;
	Slu_DEF_CFG(StatData);

	struct Stat
	{
		StatDataV<true> data;
		Position place;

		Stat() = default;
		Stat(StatDataV<true>&& data) :data(std::move(data)) {}
		Stat(const Stat&) = delete;
		Stat(Stat&&) = default;
		Stat& operator=(Stat&&) = default;
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