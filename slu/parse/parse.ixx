module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>
#include <unordered_set>
#include <format>

#include <slu/Ansi.hpp>
#include <slu/Panic.hpp>
#include <slu/ext/CppMatch.hpp>
export module slu.parse.parse;

import slu.ast.enums;
import slu.ast.small_enum_list;
import slu.ast.state;
import slu.parse.error;
import slu.parse.input;
import slu.parse.adv.block;
import slu.parse.adv.expr;
import slu.parse.adv.expr_base;
import slu.parse.adv.pat;
import slu.parse.adv.recover_from_error;
import slu.parse.adv.tbl;
import slu.parse.basic.args;
import slu.parse.basic.label;
import slu.parse.basic.mod_stat;
import slu.parse.basic.struct_stat;
import slu.parse.basic.use_stat;
import slu.parse.com.name;
import slu.parse.com.skip_space;
import slu.parse.com.str;
import slu.parse.com.tok;
import slu.parse.errors.kw;



/*

	[X] EOS handling

	[X] chunk ::= block

	[X] block ::= {stat} [retstat]

	[X] stat ::= [X] ‘;’ |
		[X] varlist ‘=’ explist |
		[X] functioncall |
		[X] label |
		[X] break |
		[X] goto Name |
		[X] do block end |
		[X] while exp do block end |
		[X] repeat block until exp |
		[X] if exp then block {elseif exp then block} [else block] end |
		[X] for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end |
		[X] for namelist in explist do block end |
		[X] function funcname funcbody |
		[X] local function Name funcbody |
		[X] local attnamelist [‘=’ explist]

	[X] attnamelist ::=  Name attrib {‘,’ Name attrib}

	[X] attrib ::= [‘<’ Name ‘>’]

	[X] retstat ::= return [explist] [‘;’]

	[X] label ::= ‘::’ Name ‘::’

	[X] funcname ::= Name {‘.’ Name} [‘:’ Name]

	[X] varlist ::= var {‘,’ var}

	[X] var ::=  Name | prefixexp ‘[’ exp ‘]’ | prefixexp ‘.’ Name

	[X] namelist ::= Name {‘,’ Name}

	[X] explist ::= exp {‘,’ exp}

	[X] exp ::=  [X]nil | [X]false | [X]true | [X]Numeral | [X]LiteralString | [X]‘...’ | [X]functiondef |
		 [X]prefixexp | [X]tableconstructor | [X]exp binop exp | [X]unop exp

	[X] prefixexp ::= var | functioncall | ‘(’ exp ‘)’

	[X] functioncall ::=  prefixexp args | prefixexp ‘:’ Name args

	[X] args ::=  ‘(’ [explist] ‘)’ | tableconstructor | LiteralString

	[X] functiondef ::= function funcbody

	[X] funcbody ::= ‘(’ [parlist] ‘)’ block end

	[X] parlist ::= namelist [‘,’ ‘...’] | ‘...’

	[X] tableconstructor ::= ‘{’ [fieldlist] ‘}’

	[X] fieldlist ::= field {fieldsep field} [fieldsep]

	[X] field ::= ‘[’ exp ‘]’ ‘=’ exp | Name ‘=’ exp | exp

	[X] fieldsep ::= ‘,’ | ‘;’

	[X] binop ::=  ‘+’ | ‘-’ | ‘*’ | ‘/’ | ‘//’ | ‘^’ | ‘%’ |
		 ‘&’ | ‘~’ | ‘|’ | ‘>>’ | ‘<<’ | ‘..’ |
		 ‘<’ | ‘<=’ | ‘>’ | ‘>=’ | ‘==’ | ‘~=’ |
		 and | or

	[X] unop ::= ‘-’ | not | ‘#’ | ‘~’

*/


namespace slu::parse
{
	export ast::SmallEnumList<ast::UnOpType> readSelfArgSpecifiers(AnyInput auto& in)
	{
		ast::SmallEnumList<ast::UnOpType> specifiers;
		while (true)
		{
			skipSpace(in);
			uint8_t r;
			switch (in.peek())
			{
			case '&':
				in.skip();
				r = 0;
				break;
			case '*':
				in.skip();
				r = uint8_t(ast::UnOpType::PTR) - uint8_t(ast::UnOpType::REF);
				break;
			default:
				return specifiers;
			}
			if (checkReadTextToken(in, "mut"))
				r += (uint8_t)ast::UnOpType::REF_MUT;
			else if (checkReadTextToken(in, "share"))
				r += (uint8_t)(ast::UnOpType::REF_SHARE);
			else if (checkReadTextToken(in, "const"))
				r += (uint8_t)(ast::UnOpType::REF_CONST);
			else
				r += (uint8_t)ast::UnOpType::REF;
			specifiers.push_back((ast::UnOpType)r);
		}
		return specifiers;
	}

	export template<AnyInput In>
	parse::Parameter readFuncParam(In& in)
	{
		parse::Parameter p;
		skipSpace(in);
		if(checkReadTextToken(in, "const"))
			p.name = in.genData.template resolveNewName<false>(readName(in));
		else
			p.name = in.genData.template resolveNewName<true>(readName(in));
		requireToken(in, "=");
		p.type = readExpr(in, false);
		return p;
	}

	export template<AnyInput In>
	FunctionInfo readFuncInfo(In& in)
	{
		/*
			funcbody ::= ‘(’ [parlist] ‘)’ block end
			parlist ::= namelist [‘,’ ‘...’] | ‘...’
		*/
		FunctionInfo ret{};

		requireToken(in, "(");

		skipSpace(in);

		char ch = in.peek();

		if (ch == '&' || ch == '*' || (ch == 's' && checkTextToken(in, "self")))
		{
			ret.selfArg.specifiers = readSelfArgSpecifiers(in);
			requireToken(in, "self");
			ret.selfArg.name = in.genData.resolveNewName<true>("self");

			skipSpace(in);

			ch = in.peek();
			if(ch == ',')
			{
				in.skip();
				skipSpace(in);
				ch = in.peek();
			}
			else if(ch != ')')
			{
				in.handleError(std::format(
					"Expected "
					LUACC_SINGLE_STRING(",") " or " 
					LUACC_SINGLE_STRING(")") " after self argument"
					"{}", errorLocStr(in)));
			}
		}
		else
			ret.selfArg.name = parse::LocalId::newEmpty();

		if (ch == '.')
		{
			throwUnexpectedVarArgs(in);
			//requireToken(in, "...");
			//ret.hasVarArgParam = true;
		}
		else if (ch != ')')
		{//must have non-empty namelist

			ret.params.emplace_back(readFuncParam(in));

			while (checkReadToken(in, ","))
			{
				if (checkReadToken(in, "..."))
				{
					throwUnexpectedVarArgs(in);
					//ret.hasVarArgParam = true;
					//break;//cant have anything after the ... arg
				}
				ret.params.emplace_back(readFuncParam(in));
			}
		}

		requireToken(in, ")");
		if (checkReadToken(in, "->"))
			ret.retType = std::make_unique<Expr>(readExpr<true>(in, false));

		return ret;
	}
	export template<AnyInput In>
	std::variant<Function, FunctionInfo> readFuncBody(In& in,std::optional<std::string> funcName)
	{
		ast::Position place = in.getLoc();
		in.genData.pushLocalScope();

		if (funcName.has_value())
			in.genData.pushScope(in.getLoc(),std::move(*funcName));
		else
			in.genData.pushAnonScope(in.getLoc());

		FunctionInfo fi = readFuncInfo(in);
		skipSpace(in);
		if (!in || (in.peek() != '{'))//no { found?
		{
			fi.local2Mp = in.genData.popLocalScope();
			in.genData.popScope(in.getLoc());//TODO: maybe add it to the func info?
			return std::move(fi);//No block, just the info
		}

		Function func = { std::move(fi) };

		try
		{
			requireToken(in, "{");
			func.block = readBlock<false>(in, func.hasVarArgParam, false);
			requireToken(in, "}");
			func.local2Mp = in.genData.popLocalScope();
		}
		catch (const ParseError&)
		{
			func.local2Mp = in.genData.popLocalScope();
			throw;
		}
		return std::move(func);
	}

	export template<bool isLoop, AnyInput In>
	Block<In> readDoOrStatOrRet(In& in, const bool allowVarArg)
	{
		if (checkReadToken(in, "{"))
		{
			return readBlockNoStartCheck<isLoop>(in, allowVarArg, true);
		}

		in.genData.pushAnonScope(in.getLoc());//readBlock also pushes!

		if (readReturn<isLoop>(in, allowVarArg))
			return in.genData.popScope(in.getLoc());
		//Basic Stat + ';'

		readStat<isLoop>(in, allowVarArg);

		Block<In> bl = in.genData.popScope(in.getLoc());

		readOptToken(in, ";");

		return bl;
	}

	export template<bool isLoop,bool BASIC, AnyInput In>
	Soe<In> readSoe(In& in, const bool allowVarArg,const bool allowArrow)
	{
		if (checkReadToken(in, "{"))
		{
			return readBlockNoStartCheck<isLoop>(in, allowVarArg, true);
		}
		RetType ret = readRetStart(in);
		if (ret != RetType::NONE)
		{
			in.genData.pushAnonScope(in.getLoc());
			readReturnAfterStart<isLoop>(in, allowVarArg,ret);
			return in.genData.popScope(in.getLoc());
		}
		if(allowArrow)
			readOptToken<false>(in, "=>");
		return std::make_unique<Expr>(readExpr<BASIC>(in, allowVarArg));
	}

	template<AnyInput In>
	void readWhereClauses(In& in, WhereClauses& itm)
	{
		if (checkReadTextToken(in, "where"))
		{
			while (true)
			{
				WhereClause& c = itm.emplace_back();
				c.var = in.genData.resolveName(readName<NameCatagory::BOUND_VAR>(in));
				requireToken(in, ":");
				c.bound = readTraitExpr(in);

				if (!checkReadToken(in, ","))
					break;
			}
		}
	}
	template<bool isLoop,AnyInput In>
	bool readTchStat(In& in, const ast::Position place, const lang::ExportData exported)
	{
		if (checkReadTextToken(in, "trait"))
		{
			StatType::Trait res;
			res.exported = exported;
			std::string name = readName(in);
			res.name = in.genData.addLocalObj(name);
			in.genData.pushScope(in.getLoc(), std::move(name));
			skipSpace(in);
			if (in.peek() == '(')
			{
				in.skip();
				res.params = readParamList<false>(in);
				skipSpace(in);
			}
			if (in.peek() == ':')
			{
				in.skip();
				res.whereSelf = readTraitExpr(in);
			}
			readWhereClauses(in,res.clauses);

			requireToken(in, "{");
			res.itms = readGlobStatList<false>(in);
			requireToken(in, "}");
			in.genData.addStat(place, std::move(res));
			return true;
		}
		return false;
	}
	template<bool isLoop, AnyInput In>
	bool readIchStat(In& in, const ast::Position place, const lang::ExportData exported, const ast::OptSafety safety,const bool hasDefer,const bool allowVarArg)
	{
		if (in.isOob(1))
			return false;
		switch (in.peekAt(1))
		{
		case 'f':
			if (safety == ast::OptSafety::DEFAULT && !hasDefer && !exported 
				&& checkReadTextToken(in, "if"))
			{ // if exp then block {elseif exp then block} [else block] end
				in.genData.addStat(place,
					readIfCond<isLoop, false, false>(
						in, allowVarArg
					));
				return true;
			}
			break;
		case 'm':
			if (checkReadTextToken(in, "impl"))
			{
				if (safety == ast::OptSafety::SAFE)
					throwUnexpectedSafety(in, place);

				StatType::Impl res;
				res.exported = exported;
				res.deferChecking = hasDefer;
				res.isUnsafe = safety == ast::OptSafety::UNSAFE;
				skipSpace(in);
				if (in.peek() == '(')
				{
					in.skip();
					res.params = readParamList<false>(in);
				}
				TraitExpr traitOrType = readTraitExpr(in);
				if (checkReadTextToken(in, "for"))
				{
					res.forTrait = std::move(traitOrType);
					res.type = readExpr<true>(in, false);
				}
				else
				{
					if (traitOrType.traitCombo.size() != 1)
						throwExpectedTypeExpr(in);
					res.type = std::move(traitOrType.traitCombo[0]);
				}
				readWhereClauses(in, res.clauses);

				requireToken(in, "{");
				res.code = readGlobStatList<true>(in);
				requireToken(in, "}");
				in.genData.addStat(place, std::move(res));
				return true;
			}
			break;
		}
		return false;
	}

	template<bool isLoop,AnyInput In>
	bool readUchStat(In& in, const ast::Position place, const lang::ExportData exported)
	{
		if (in.isOob(2))
			return false;
		switch (in.peekAt(2))
		{
		case 'e':
			if (readUseStat(in, place, exported))
				return true;
			break;
		case 's':
			if (checkReadTextToken(in, "unsafe"))
			{
				skipSpace(in);
				switch (in.peek())
				{
				case 'i':
					if (readIchStat<isLoop>(in, place, exported, ast::OptSafety::UNSAFE,false, false))
						return true;
					break;
				case 'd'://defer impl?
					if (checkReadTextToken(in, "defer"))
					{
						skipSpace(in);
						if (in.peek() == 'i')
						{
							if (readIchStat<isLoop>(in, place, exported, ast::OptSafety::UNSAFE, true, false))
								return true;
						}
						throwExpectedImplAfterDefer(in);
					}
					break;
				case 'f':
					if (readFchStat<isLoop>(in, place, exported, ast::OptSafety::UNSAFE, false))
						return true;
					break;
				case '{':
				{
					in.skip();
					in.genData.pushUnsafe();
					StatType::UnsafeBlock<In> res = { readStatList<isLoop>(in, false,false).first };
					requireToken(in, "}");
					in.genData.popSafety();
					in.genData.addStat(place, std::move(res));
					return true;
				}
				case 'e':
					if (!exported && readEchStat<isLoop>(in, place, ast::OptSafety::UNSAFE, false))
						return true;
					break;
				case 't'://unsafe traits?
				default:
					break;
				}
				throwExpectedUnsafeable(in);
			}
			break;
		case 'i':
			if (checkReadTextToken(in, "union"))
			{
				readStructStat<StatType::Union, true>(in, place, exported);
				return true;
			}
		default:
			break;
		}
		return false;
	}

	template<bool isLoop, AnyInput In>
	bool readEchStat(In& in, const ast::Position place, const ast::OptSafety safety, const bool allowVarArg)
	{
		if(checkReadTextToken(in,"extern"))
		{
			skipSpace(in);
			std::string abi = readStringLiteral(in, in.peek());
			ast::Position abiEnd = in.getLoc();
			skipSpace(in);
			if (in.peek() == '{')
			{
				in.skip();
				StatType::ExternBlock<In> res{};

				res.safety = safety;
				res.abi = std::move(abi);
				res.abiEnd = abiEnd;
				res.stats = readStatList<isLoop>(in, allowVarArg,false).first;
				requireToken(in, "}");

				in.genData.addStat(place, std::move(res));
				return true;
			}
			//TODO: [safety] extern "" fn

			throwExpectedExternable(in);
		}
		return false;
	}
	template<bool isLoop, AnyInput In>
	bool readFchStat(In& in, const ast::Position place, const lang::ExportData exported,const ast::OptSafety safety, const bool allowVarArg)
	{
		if (in.isOob(1))
			return false;
		const char ch2 = in.peekAt(1);
		switch (ch2)
		{
		case 'n':
			if (checkReadTextToken(in, "fn"))
			{
				readFunctionStat<isLoop, StatType::Fn, StatType::FnDecl<In>>(
					in, place, allowVarArg, exported, safety
				);
				return true;
			}
			break;
		case 'u':
			if (checkReadTextToken(in, "function"))
			{
				readFunctionStat<isLoop, StatType::Function, StatType::FunctionDecl<In>>(
					in, place, allowVarArg, exported, safety
				);
				return true;
			}
			break;
		case 'o':
			if (exported || safety != ast::OptSafety::DEFAULT)
				break;
			if (checkReadTextToken(in, "for"))
			{
				/*
				 for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end |
				 for namelist in explist do block end |
				*/

				PatV<true, true> names;
				skipSpace(in);
				names = readPat<true>(in, true);

				// 'for' pat 'in' exp '{' block '}' 

				StatType::ForIn<In> res{};
				res.varNames = std::move(names);

				requireToken(in, "in");
				res.exprs = readExpr<true>(in, allowVarArg);


				res.bl = readDoOrStatOrRet<true>(in, allowVarArg);

				in.genData.addStat(place, std::move(res));
				return true;
			}
			break;
		default:
			break;
		}
		return false;
	}
	template<bool isLoop, AnyInput In>
	bool readLchStat(In& in, const ast::Position place, const lang::ExportData exported, const bool allowVarArg)
	{
		if (in.isOob(1))
			return false;

		const char ch2 = in.peekAt(1);
		switch (ch2)
		{
		case 'e':
			if (checkReadTextToken(in, "let"))
			{
				readVarStat<true, isLoop, StatType::Let<In>>(in, place, allowVarArg, exported);
				return true;
			}
			break;
		case 'o':
			if (checkReadTextToken(in, "local"))
			{
				// Local Variable
				readVarStat<true, isLoop, StatType::Local<In>>(in, place, allowVarArg, exported);
				return true;
			}
			break;
		default:
			break;
		}
		return false;
	}

	template<bool isLoop, AnyInput In>
	bool readCchStat(In& in, const ast::Position place, const lang::ExportData exported, const bool allowVarArg)
	{
		if (in.isOob(2))
			return false;

		const char ch2 = in.peekAt(2);
		switch (ch2)
		{
		case 'm':
			if (checkReadTextToken(in, "comptime"))
				throw 333;
			//TODO
			break;
		case 'n':
			if (checkReadTextToken(in, "const"))
			{
				readVarStat<false,isLoop, StatType::Const<In>>(in, place, allowVarArg, exported);
				return true;
			}
			break;
		default:
			break;
		}
		return false;
	}


	template<bool isLoop,AnyInput In>
	bool readSchStat(In& in, const ast::Position place, const lang::ExportData exported)
	{
		if (in.isOob(1))
			return false;

		const char ch2 = in.peekAt(1);
		switch (ch2)
		{
		case 'a':
			if (checkReadTextToken(in, "safe"))
			{
				skipSpace(in);
				switch (in.peek())
				{
				case 'e':
					if(!exported && readEchStat<isLoop>(in,place, ast::OptSafety::SAFE, false))
						return true;
					break;
				case 'f':
					if (readFchStat<isLoop>(in, place, exported, ast::OptSafety::SAFE, false))
						return true;
					break;
				default:
					break;
				}
				throwExpectedSafeable(in);
			}
			break;
		case 't':
			if (checkReadTextToken(in, "struct"))
			{
				//TODO: `struct fn`
				readStructStat<StatType::Struct,false>(in, place, exported);
				return true;
			}
			break;
		default:
			break;
		}
		return false;
	}

	template<bool isLoop,class StatT,class DeclStatT, AnyInput In>
	void readFunctionStat(In& in, 
		const ast::Position place, const bool allowVarArg, 
		const lang::ExportData exported, const ast::OptSafety safety)
	{
		StatT res{};
		std::string name;//moved @ readFuncBody
		name = readName(in);
		res.name = in.genData.addLocalObj(name);
		res.place = in.getLoc();

		try
		{
			auto fun = readFuncBody(in,std::move(name));
			if(ezmatch(std::move(fun))(
				varcase(Function&&) {
					res.func = std::move(var);
					return false;
				},
				varcase(FunctionInfo&&) {
					DeclStatT declRes{ std::move(var) };
					declRes.name = res.name;
					declRes.place = res.place;
					declRes.exported = exported;
					declRes.safety = safety;

					in.genData.addStat(place, std::move(declRes));
					return true;
				}
			))
				return;//Stat was added

		}
		catch (const ParseError& e)
		{
			in.handleError(e.m);
			throw ErrorWhileContext(std::format(
				"In " LC_function " " LUACC_SINGLE_STRING("{}") " at {}",
				in.genData.asSv(res.name), errorLocStr(in, res.place)
			));
		}

		res.exported = exported;
		res.func.safety = safety;

		return in.genData.addStat(place, std::move(res));
	}
	//TODO: handle basic (in basic expressions, if expressions can only have basic expressions)
	export template<bool isLoop,bool forExpr,bool BASIC, AnyInput In>
	auto readIfCond(In& in, const bool allowVarArg)
	{
		BaseIfCond<In,forExpr> res{};

		res.cond = mayBoxFrom<forExpr>(readBasicExpr(in, allowVarArg));

		res.bl = mayBoxFrom<forExpr>(readSoe<isLoop, BASIC>(in, allowVarArg,true));

		while (checkReadTextToken(in, "else"))
		{
			if (checkReadTextToken(in, "if"))
			{
				Expr elExpr = readBasicExpr(in, allowVarArg);
				Soe<In> elBlock = readSoe<isLoop, BASIC>(in, allowVarArg,true);

				res.elseIfs.emplace_back(std::move(elExpr), std::move(elBlock));
				continue;
			}

			res.elseBlock = mayBoxFrom<forExpr>(readSoe<isLoop, BASIC>(in, allowVarArg,false));
			break;
		}
		return res;
	}
	template<bool isLocal, bool isLoop, class StatT, AnyInput In >
	void readVarStat(In& in, const ast::Position place, const bool allowVarArg, const lang::ExportData exported)
	{
		StatT res;
		skipSpace(in);
		if constexpr (!isLocal)
			in.genData.pushLocalScope();
		res.names = readPat<isLocal>(in, true);
		res.exported = exported;

		if (checkReadToken(in, "="))
		{// [‘=’ explist]
			res.exprs = readExprList(in, allowVarArg);
		}
		if constexpr (!isLocal)
			res.local2Mp = in.genData.popLocalScope();
		return in.genData.addStat(place, std::move(res));
	}

	export template<bool isLoop, AnyInput In>
	void readStat(In& in,const bool allowVarArg)
	{
		/*
		 varlist ‘=’ explist |
		 functioncall |
		*/

		skipSpace(in);

		const ast::Position place = in.getLoc();
		Stat ret;

		const char firstChar = in.peek();
		switch (firstChar)
		{
		case ';':
			in.skip();
			return in.genData.addStat(place, StatType::Semicol{});
		case ':'://may be label
			if (in.peekAt(1) == '>')
				break;//Not a label
			return readLabel(in, place);

		case 'f'://for?, function?, fn?
			if(readFchStat<isLoop>(in, place, false, ast::OptSafety::DEFAULT, allowVarArg))
				return;
			break;
		case 'l'://local?
			if (readLchStat<isLoop>(in, place, false, allowVarArg))
				return;

			break;
		case 'c'://const comptime?
			if (readCchStat<isLoop>(in, place, false, allowVarArg))
				return;
			break;
		case '{':// ‘{’ block ‘}’
			in.skip();//Skip ‘{’
			return in.genData.addStat(place,
				StatType::Block<In>(readBlockNoStartCheck<isLoop>(in, allowVarArg, true))
			);
			break;
		case 'd'://do?
			if (checkReadTextToken(in, "drop"))
			{
				return in.genData.addStat(place,
					StatType::Drop<In>(readExpr(in, allowVarArg))
				);
			}
			if (checkReadTextToken(in, "defer"))
			{
				skipSpace(in);
				if(in.peek() == 'i')
				{
					if (readIchStat<isLoop>(in, place, false, ast::OptSafety::DEFAULT, true, allowVarArg))
						return;
				}
				throwExpectedImplAfterDefer(in);
			}
			break;
		case 'g'://goto?
			if (checkReadTextToken(in, "goto"))//goto Name
			{
				return in.genData.addStat(place,
					StatType::Goto<In>(in.genData.resolveName(readName(in)))
				);
			}
			break;
		case 'w'://while?
			if (checkReadTextToken(in, "while"))
			{ // while exp do block end

				Expr expr = readBasicExpr(in,allowVarArg);

				Block<In> bl = readDoOrStatOrRet<true>(in,allowVarArg);
				return in.genData.addStat(place, 
					StatType::While<In>(std::move(expr), std::move(bl))
				);
			}
			break;
		case 'r'://repeat?
			if (checkReadTextToken(in, "repeat"))
			{ // repeat block until exp
				Block<In> bl;
				bl = readDoOrStatOrRet<true>(in, allowVarArg);
				requireToken(in, "until");
				Expr expr = readExpr(in,allowVarArg);

				return in.genData.addStat(place, 
					StatType::RepeatUntil<In>({ std::move(expr), std::move(bl) })
				);
			}
			break;
		case 'i'://if? impl?
			if (readIchStat<isLoop>(in, place, false, ast::OptSafety::DEFAULT,false, allowVarArg))
				return;
			break;

			//Slu
		case 'e'://ex ...?
			if (checkReadTextToken(in, "ex"))
			{
				skipSpace(in);
				switch (in.peek())
				{
				case 'f'://fn? function?
					if (readFchStat<isLoop>(in, place, true, ast::OptSafety::DEFAULT, allowVarArg))
						return;
					break;
				case 't'://trait?
					if (readTchStat<isLoop>(in, place, true))
						return;
					break;
				case 'i'://impl?
					if (readIchStat<isLoop>(in, place, true, ast::OptSafety::DEFAULT, false, allowVarArg))
						return;
					break;
				case 'd'://defer impl?
					if (checkReadTextToken(in, "defer"))
					{
						skipSpace(in);
						if (in.peek() == 'i')
						{
							if (readIchStat<isLoop>(in, place, true, ast::OptSafety::DEFAULT, true, allowVarArg))
								return;
						}
						throwExpectedImplAfterDefer(in);
					}
					break;
				case 'l'://let? local?
					if (readLchStat<isLoop>(in, place, true, allowVarArg))
						return;
					break;
				case 'c'://const? comptime?
					if (readCchStat<isLoop>(in, place, true, allowVarArg))
						return;
					break;
				case 'u'://use? unsafe?
					if (readUchStat<isLoop>(in, place, true))
						return;
					break;
				case 's'://safe? struct?
					if (readSchStat<isLoop>(in, place, true))
						return;
					break;
				case 'm'://mod?
					if (readModStat(in, place, true))
						return;
					break;
				default:
					break;
				}
				throwExpectedExportable(in);
			}
			else if (readEchStat<isLoop>(in, place, ast::OptSafety::DEFAULT, allowVarArg))
				return;
			
			break;
		case 's'://safe? struct?
			if (readSchStat<isLoop>(in, place, false))
				return;
			break;
		case 'u'://use? unsafe?
			if (readUchStat<isLoop>(in, place, false))
				return;
			break;
		case 'm'://mod?
			if (readModStat(in, place, false))
				return;
			break;
		case 't'://trait?
			if (readTchStat<isLoop>(in, place, false))
				return;
			break;
		default://none of the above...
			break;
		}

		in.genData.addStat(place, 
			parsePrefixExprVar<StatData<In>,false>(
				in,allowVarArg, firstChar
		));
	}

	/**
	 * @throws slu::parse::ParseFailError
	 */
	export template<AnyInput In>
	ParsedFile parseFile(In& in)
	{
		try
		{
			ParsedFile res;
			auto [sl, mp] = readStatList<false>(in, false, true);
			res.code = std::move(sl);
			res.mp = mp;

			Slu_assert(in.genData.scopes.empty());

			if (in.hasError())
			{// Skip eof, as one of the errors might have caused that.
				throw ParseFailError();
			}

			skipSpace(in);
			if (in)
			{
				throw UnexpectedCharacterError(std::format(
					"Expected end of stream"
					", found " LUACC_SINGLE_STRING("{}")
					"{}"
					, in.peek(), errorLocStr(in)));
			}
			return  std::move(res);
		}
		catch (const BasicParseError& e)
		{
			in.handleError(e.m);
			in.genData.scopes.clear();
			throw ParseFailError();
		}
	}
}