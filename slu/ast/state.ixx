module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <optional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

export module slu.ast.state;

import slu.ast.enums;
import slu.ast.pos;
import slu.ast.small_enum_list;
import slu.ast.state_decls;
import slu.ast.type;
import slu.lang.basic_state;
import slu.parse.input;

namespace slu::parse
{
#define Slu_DEF_CFG(_Name) export template<class CfgT> using _Name = _Name ## V<true>
#define Slu_DEF_CFG2(_Name,_ArgName) export template<class CfgT,bool _ArgName> using _Name =_Name ## V<true, _ArgName>

	namespace FieldType { export using NONE = std::monostate; }

	export template<bool isSlu>
	using FieldV = std::variant<
		FieldType::NONE,// Here, so variant has a default value (DO NOT USE)

		FieldType::Expr2Expr, // "'[' exp ']' = exp"
		FieldType::Name2Expr, // "Name = exp"
		FieldType::Expr       // "exp"
	>;
	Slu_DEF_CFG(Field);


	// ‘{’ [fieldlist] ‘}’
	export template<bool isSlu>
	using TableV = std::vector<FieldV<isSlu>>;
	Slu_DEF_CFG(Table);

	export template<bool isSlu>
	using StatListV = std::vector<Stat>;
	Slu_DEF_CFG(StatList);


	export struct LocalId
	{
		size_t v = SIZE_MAX;
		constexpr bool empty() const { return v == SIZE_MAX; }
		constexpr static parse::LocalId newEmpty() { return parse::LocalId(); }
	};
	export template<bool isSlu, bool isLocal>
	using LocalOrNameV = Sel<isLocal, lang::MpItmId, parse::LocalId>;
	Slu_DEF_CFG2(LocalOrName, isLocal);
	export template<bool isSlu>
	struct LocalsV
	{
		std::vector<lang::MpItmId> names;
		std::vector<parse::ResolvedType> types;//Empty until type checking.
	};
	Slu_DEF_CFG(Locals);

	export template<bool isSlu>
	using DynLocalOrNameV = std::variant<lang::MpItmId, parse::LocalId>;

	export enum class RetType : uint8_t
	{
		NONE,
		RETURN,
		BREAK,
		CONTINUE
	};

	export template<bool isSlu>
	struct BlockV
	{
		StatListV<isSlu> statList;
		ExprList retExprs;// May contain 0 elements

		lang::ModPathId mp;

		ast::Position start;
		ast::Position end;

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
		export template<bool isSlu>
		using BlockV = BlockV<isSlu>;
		Slu_DEF_CFG(Block);

		export using Expr = BoxExpr;
	}
	export template<bool isSlu>
	using SoeV = std::variant<
		SoeType::BlockV<isSlu>,
		SoeType::Expr
	>;
	Slu_DEF_CFG(Soe);

	export template<bool isSlu> using SoeOrBlockV = Sel<isSlu, BlockV<isSlu>, SoeV<isSlu>>;
	Slu_DEF_CFG(SoeOrBlock);

	export template<bool isSlu> using SoeBoxOrBlockV = Sel<isSlu, BlockV<isSlu>, std::unique_ptr<SoeV<isSlu>>>;
	Slu_DEF_CFG(SoeBoxOrBlock);

	namespace ArgsType
	{
		export using parse::ExprList;

		export using parse::TableV;
		export using parse::Table;

		export struct String { std::string v; ast::Position end; };// "LiteralString"
	};
	export using Args = std::variant<
		ArgsType::ExprList,
		ArgsType::TableV<true>,
		ArgsType::String
	>;


	export template<bool boxed>
	struct ExprUserExpr {
		MayBox<boxed, Expr> v;
		parse::ResolvedType ty;
	};

	export template<bool boxed> // exp args
	struct Call : ExprUserExpr<boxed> {
		Args args;
	};
	export template<bool boxed> // exp "." Name args
	struct SelfCall : ExprUserExpr<boxed>
	{
		Args args;
		lang::MpItmId method;//may be unresolved, or itm=trait-fn, or itm=fn
	};

	namespace ExprType
	{
		export struct MpRoot {};		// ":>"
		export using Local = parse::LocalId;
		export template<bool isSlu>
		using GlobalV = lang::MpItmId;
		Slu_DEF_CFG(Global);

		export template<bool isSlu> // "(" exp ")"
		using ParensV = BoxExpr;
		Slu_DEF_CFG(Parens);

		export struct Deref : ExprUserExpr<true> {};// exp ".*"

		export struct Index : ExprUserExpr<true> {// exp "[" exp "]"
			MayBox<true,Expr> idx;
		};

		export template<bool isSlu> // exp "." Name
		struct FieldV : ExprUserExpr<true> {
			lang::PoolString field;
		};
		Slu_DEF_CFG(Field);

		export using Call = parse::Call<true>;
		export using SelfCall = parse::SelfCall<true>;
	}

	export struct TupleName
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

	export template<class T>
	concept Any128BitInt =
		std::same_as<T, ExprType::P128>
		|| std::same_as<T, ExprType::M128>;
	export template<class T>
	concept Any64BitInt =
		std::same_as<T, ExprType::U64>
		|| std::same_as<T, ExprType::I64>;

	//Slu
	export using Lifetime = std::vector<lang::MpItmId>;
	export struct UnOpItem
	{
		Lifetime life;
		ast::UnOpType type;
	};

	export struct TraitExpr
	{
		std::vector<Expr> traitCombo;
		ast::Position place;
	};
	export using TypePrefix = std::vector<UnOpItem>;


	//Common

	export template<bool isLocal>
	struct Parameter
	{
		LocalOrNameV<true,isLocal> name;
		Expr type;
	};

	export template<bool isLocal>
	using ParamList = std::vector<Parameter<isLocal>>;

	export struct FunctionInfo
	{
		std::string abi;
		LocalsV<true> local2Mp;
		ParamList<true> params;
		std::optional<BoxExpr> retType;
		bool hasVarArgParam = false;// do params end with '...'
		ast::OptSafety safety = ast::OptSafety::DEFAULT;
	};
	export struct Function : FunctionInfo
	{
		BlockV<true> block;
	};



	export template<bool isSlu, bool boxIt>
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
		export using Nil = std::monostate;	// "nil"
		export struct False {};				// "false"
		export struct True {};				// "true"
		export struct VarArgs {};			// "..."

		export using parse::Function;
		
		export using parse::TableV;
		export using parse::Table;

		//unOps is always empty for this type
		export template<bool isSlu>
		struct MultiOpV
		{
			BoxExpr first;
			std::vector<std::pair<ast::BinOpType, Expr>> extra;//size>=1
		};      // "exp binop exp"
		Slu_DEF_CFG(MultiOp);

		export template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, true>;
		Slu_DEF_CFG(IfCond);

		export using parse::Lifetime;	// " '/' var" {'/' var"}
		export using parse::TraitExpr;

		export struct PatTypePrefix {};

		export struct Infer {};
		export struct Struct
		{
			TableV<true> fields;
		};
		export struct Union
		{
			TableV<true> fields;
		};
		export struct FnType
		{
			BoxExpr argType;
			BoxExpr retType;
			ast::OptSafety safety = ast::OptSafety::DEFAULT;
		};
		export struct Dyn {
			parse::TraitExpr expr;
		};
		export struct Impl {
			parse::TraitExpr expr;
		};
		export struct Err
		{
			BoxExpr err;
		};
	}

	export template<bool isSlu>
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
		ExprType::Err,
		ExprType::FnType
	> ;
	Slu_DEF_CFG(ExprData);



	export struct Expr
	{
		ExprDataV<true> data;
		ast::Position place;
		std::vector<UnOpItem> unOps;//TODO: for lua, use small op list

		ast::SmallEnumList<ast::PostUnOpType> postUnOps;

		bool isBasicStruct() const {
			if (!this->unOps.empty() || !this->postUnOps.empty())
				return false;
			return std::holds_alternative<ExprType::TableV<true>>(this->data);
		}

		Expr() = default;
		Expr(ExprDataV<true>&& data) 
			:data(std::move(data)) {}
		Expr(ExprDataV<true>&& data, ast::Position place) 
			:data(std::move(data)), place(place) {}
		Expr(ExprDataV<true>&& data, ast::Position place, std::vector<UnOpItem>&& unOps) 
			:data(std::move(data)), place(place), unOps(std::move(unOps)) {}

		Expr(const Expr&) = delete;
		Expr(Expr&&) = default;
		Expr& operator=(Expr&&) = default;
	};

	//Slu


	// match patterns

	export using NdPat = Expr;

	namespace DestrSpecType
	{
		export using Spat = parse::NdPat;
		export using Prefix = TypePrefix;
	}
	export using DestrSpec = std::variant<
		DestrSpecType::Spat,
		DestrSpecType::Prefix
	>;
	namespace DestrPatType
	{
		export template<bool isSlu, bool isLocal>
		using AnyV = LocalOrNameV<isSlu,isLocal>;
		Slu_DEF_CFG2(Any, isLocal);

		export template<bool isSlu, bool isLocal> struct FieldsV;
		Slu_DEF_CFG2(Fields, isLocal);
		export template<bool isSlu, bool isLocal> struct ListV;
		Slu_DEF_CFG2(List, isLocal);

		export template<bool isSlu, bool isLocal>struct NameV;
		Slu_DEF_CFG2(Name, isLocal);
		export template<bool isSlu, bool isLocal>struct NameRestrictV;
		Slu_DEF_CFG2(NameRestrict, isLocal);
	}

	namespace PatType
	{
		//x or y or z
		export template<bool isSlu>
		using SimpleV = NdPat;
		Slu_DEF_CFG(Simple);

		export template<bool isSlu, bool isLocal>
		using DestrAnyV = DestrPatType::AnyV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrAny, isLocal);

		export template<bool isSlu,bool isLocal>
		using DestrFieldsV = DestrPatType::FieldsV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrFields, isLocal);
		export template<bool isSlu, bool isLocal>
		using DestrListV = DestrPatType::ListV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrList, isLocal);

		export template<bool isSlu, bool isLocal>
		using DestrNameV = DestrPatType::NameV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrName, isLocal);

		export template<bool isSlu, bool isLocal>
		using DestrNameRestrictV = DestrPatType::NameRestrictV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrNameRestrict, isLocal);
	}
	export template<bool isLocal,typename T>
	concept AnyCompoundDestr =
		std::same_as<std::remove_cv_t<T>, DestrPatType::FieldsV<true, isLocal>>
		|| std::same_as<std::remove_cv_t<T>, DestrPatType::ListV<true, isLocal>>;


	export template<bool isSlu,bool isLocal>
	using PatV = std::variant<
		PatType::DestrAnyV<isSlu, isLocal>,

		PatType::SimpleV<isSlu>,

		PatType::DestrFieldsV<isSlu, isLocal>,
		PatType::DestrListV<isSlu, isLocal>,

		PatType::DestrNameV<isSlu, isLocal>,
		PatType::DestrNameRestrictV<isSlu, isLocal>
	>;
	Slu_DEF_CFG2(Pat, isLocal);

	export template<bool isSlu, bool isLocal>
	struct DestrFieldV
	{
		lang::PoolString name;// |(...)| thingy
		PatV<isSlu,isLocal> pat;//May be any type of pattern
	};
	Slu_DEF_CFG2(DestrField,isLocal);
	namespace DestrPatType
	{
		export template<bool isSlu, bool isLocal>
		struct FieldsV
		{
			DestrSpec spec;
			bool extraFields : 1 = false;
			std::vector<DestrFieldV<isSlu, isLocal>> items;
			LocalOrNameV<isSlu, isLocal> name;//May be synthetic
		};
		export template<bool isSlu, bool isLocal>
		struct ListV
		{
			DestrSpec spec;
			bool extraFields : 1 = false;
			std::vector<PatV<isSlu, isLocal>> items;
			LocalOrNameV<isSlu,isLocal> name;//May be synthetic
		};

		export template<bool isSlu, bool isLocal>
		struct NameV
		{
			LocalOrNameV<isSlu, isLocal> name;
			DestrSpec spec;
		};
		export template<bool isSlu, bool isLocal>
		struct NameRestrictV : NameV<isSlu, isLocal>
		{
			NdPat restriction;
		};
	}

	export template<bool isLocal>
	struct ___PatHack : PatV<true, isLocal> {};

	//Common

	namespace FieldType
	{
		export struct Expr2Expr { parse::Expr idx; parse::Expr v; };		// "‘[’ exp ‘]’ ‘=’ exp"
		export struct Name2Expr { lang::MpItmId idx; parse::Expr v; };	// "Name ‘=’ exp"
	}

	export template<bool isSlu>
	struct AttribNameV
	{
		lang::MpItmId name;
		std::string attrib;//empty -> no attrib
	};
	Slu_DEF_CFG(AttribName);
	export template<bool isSlu>
	using AttribNameListV = std::vector<AttribNameV<isSlu>>;
	Slu_DEF_CFG(AttribNameList);
	export template<bool isSlu>
	using NameListV = std::vector<lang::MpItmId>;
	Slu_DEF_CFG(NameList);

	namespace UseVariantType
	{
		export using EVERYTHING_INSIDE = std::monostate;//use x::*;
		export struct IMPORT { lang::MpItmId name; };// use x::y; //name is inside this mp, base is the imported path.
		export using AS_NAME = lang::MpItmId;//use x as y;
		export using LIST_OF_STUFF = std::vector<lang::MpItmId>;//use x::{self, ...}
	}
	export using UseVariant = std::variant<
		UseVariantType::EVERYTHING_INSIDE,
		UseVariantType::AS_NAME,
		UseVariantType::IMPORT,
		UseVariantType::LIST_OF_STUFF
	>;

	export struct StructBase
	{
		ParamList<true> params;
		LocalsV<true> local2Mp;
		TableV<true> type;
		lang::MpItmId name;
		lang::ExportData exported = false;
	};
	export template<bool isSlu, bool isLocal>
	struct MaybeLocalsV {
		LocalsV<isSlu> local2Mp;
	};
	export template<bool isSlu>
	struct MaybeLocalsV<isSlu,true> {};

	export template<bool isSlu, bool isLocal>
	struct VarStatBaseV : MaybeLocalsV<isSlu, isLocal>
	{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
		PatV<true, isLocal> names;
		ExprList exprs;
		lang::ExportData exported = false;
	};

	export struct WhereClause
	{
		TraitExpr bound;
		lang::MpItmId var;
	};
	export using WhereClauses = std::vector<WhereClause>;

	namespace StatType
	{
		export using Semicol = std::monostate;	// ";"

		export template<bool isSlu>
		struct AssignV { std::vector<ExprDataV<isSlu>> vars; ExprList exprs; };// "varlist = explist" //e.size must be > 0
		Slu_DEF_CFG(Assign);

		export using Call = parse::Call<false>;
		export using SelfCall = parse::SelfCall<false>;


		export template<bool isSlu>
		struct LabelV { lang::MpItmId v; };		// "label"
		Slu_DEF_CFG(Label);
		export template<bool isSlu>
		struct GotoV { lang::MpItmId v; };			// "goto Name"
		Slu_DEF_CFG(Goto);

		export using parse::BlockV;// "do block end"
		export using parse::Block;

		export template<bool isSlu>
		struct WhileV { Expr cond; BlockV<isSlu> bl; };		// "while exp do block end"
		Slu_DEF_CFG(While);

		export template<bool isSlu>
		struct RepeatUntilV :WhileV<isSlu> {};						// "repeat block until exp"
		Slu_DEF_CFG(RepeatUntil);

		// "if exp then block {elseif exp then block} [else block] end"
		export template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, false>;
		Slu_DEF_CFG(IfCond);

		// "for namelist in explist do block end"
		export template<bool isSlu>
		struct ForInV
		{
			Sel<isSlu, NameListV<isSlu>, PatV<true, true>> varNames;
			Sel<isSlu, ExprList, Expr> exprs;//size must be > 0
			BlockV<isSlu> bl;
		};
		Slu_DEF_CFG(ForIn);

		export template<bool isSlu>
		struct FuncDefBase
		{// "function funcname funcbody"    
			ast::Position place;//Right after func-name
			lang::MpItmId name; // name may contain dots, 1 colon if !isSlu
			Function func;
		};
		export struct Function : FuncDefBase<true> {
			lang::ExportData exported = false;
		};

		export struct Fn : Function {};

		export template<bool isSlu>
		struct FunctionDeclV : FunctionInfo
		{
			ast::Position place;//Right before func-name
			lang::MpItmId name;
			lang::ExportData exported = false;
		};
		Slu_DEF_CFG(FunctionDecl);

		export template<bool isSlu>
		struct FnDeclV : FunctionDeclV<isSlu> {};
		Slu_DEF_CFG(FnDecl);

		export template<bool isSlu>
		using LocalV = VarStatBaseV<isSlu,true>;
		Slu_DEF_CFG(Local);

		// Slu

		export template<bool isSlu>
		struct LetV : LocalV<isSlu> {};
		Slu_DEF_CFG(Let);

		export template<bool isSlu>
		using ConstV = VarStatBaseV<isSlu, false>;
		Slu_DEF_CFG(Const);

		export struct CanonicLocal
		{
			ResolvedType type;
			parse::LocalId name;
			Expr value;
			lang::ExportData exported = false;
		};
		export struct CanonicGlobal
		{
			ResolvedType type;
			LocalsV<true> local2Mp;
			lang::MpItmId name;
			Expr value;
			lang::ExportData exported = false;
		};


		export struct Struct : StructBase {};
		export struct Union : StructBase {};

		export template<bool isSlu>
		struct ExternBlockV
		{
			StatListV<isSlu> stats;
			std::string abi;
			ast::Position abiEnd;
			ast::OptSafety safety = ast::OptSafety::DEFAULT;
		};
		Slu_DEF_CFG(ExternBlock);

		export template<bool isSlu>
		struct UnsafeBlockV {
			StatListV<isSlu> stats;
		};	// "unsafe {...}"
		Slu_DEF_CFG(UnsafeBlock);

		export struct UnsafeLabel {};
		export struct SafeLabel {};

		export struct Trait
		{
			WhereClauses clauses;
			lang::MpItmId name;
			ParamList<false> params;
			StatListV<true> itms;
			std::optional<TraitExpr> whereSelf;
			lang::ExportData exported = false;
		};
		export struct Impl
		{
			WhereClauses clauses;
			ParamList<false> params;
			std::optional<TraitExpr> forTrait;
			Expr type;
			StatListV<true> code;
			lang::ExportData exported: 1 = false;
			bool deferChecking : 1 = false;
			bool isUnsafe	   : 1 = false;
		};

		export template<bool isSlu>
		struct DropV
		{
			Expr expr;
		};
		Slu_DEF_CFG(Drop);


		export struct Use
		{
			lang::MpItmId base;//the aliased/imported thing, or modpath base
			UseVariant useVariant;
			lang::ExportData exported = false;
		};

		export template<bool isSlu>
		struct ModV
		{
			lang::MpItmId name;
			lang::ExportData exported = false;
		};
		Slu_DEF_CFG(Mod);

		export template<bool isSlu>
		struct ModAsV : ModV<isSlu>
		{
			StatListV<isSlu> code;
		};
		Slu_DEF_CFG(ModAs);
	};

	export template<bool isSlu>
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

	export struct Stat
	{
		StatDataV<true> data;
		ast::Position place;

		Stat() = default;
		Stat(StatDataV<true>&& data) :data(std::move(data)) {}
		Stat(const Stat&) = delete;
		Stat(Stat&&) = default;
		Stat& operator=(Stat&&) = default;
	};

	export struct ParsedFile
	{
		StatListV<true> code;
		lang::ModPathId mp;
	};
}