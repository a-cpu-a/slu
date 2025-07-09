/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/parser/Input.hpp>
#include <slu/parser/State.hpp>

#include "basic/ReadArgs.hpp"
#include "basic/ReadMiscNames.hpp"
#include "basic/ReadBasicStats.hpp"
#include "basic/ReadModStat.hpp"
#include "basic/ReadUseStat.hpp"
#include "basic/ReadStructStat.hpp"
#include "adv/ReadName.hpp"
#include "adv/RequireToken.hpp"
#include "adv/SkipSpace.hpp"
#include "adv/ReadExpr.hpp"
#include "adv/ReadBlock.hpp"
#include "adv/ReadStringLiteral.hpp"
#include "adv/ReadTable.hpp"
#include "adv/RecoverFromError.hpp"
#include "adv/ReadPat.hpp"
#include "errors/KwErrors.h"



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
	template<AnyInput In>
	inline Parameter<In> readFuncParam(In& in)
	{
		Parameter<In> p;
		if constexpr (In::settings() & sluSyn)
		{
			skipSpace(in);
			p.name = readPat<true>(in, true);
		}
		else
			p.name = in.genData.resolveUnknown(readName(in));
		
		return p;
	}

	template<AnyInput In>
	inline FunctionInfo<In> readFuncInfo(In& in)
	{
		/*
			funcbody ::= ‘(’ [parlist] ‘)’ block end
			parlist ::= namelist [‘,’ ‘...’] | ‘...’
		*/
		FunctionInfo<In> ret{};

		requireToken(in, "(");

		skipSpace(in);

		const char ch = in.peek();

		if (ch == '.')
		{
			if constexpr (In::settings() & sluSyn)
				throwUnexpectedVarArgs(in);
			requireToken(in, "...");
			ret.hasVarArgParam = true;
		}
		else if (ch != ')')
		{//must have non-empty namelist
			ret.params.emplace_back(readFuncParam(in));

			while (checkReadToken(in, ","))
			{
				if (checkReadToken(in, "..."))
				{
					if constexpr (In::settings() & sluSyn)
						throwUnexpectedVarArgs(in);
					ret.hasVarArgParam = true;
					break;//cant have anything after the ... arg
				}
				ret.params.emplace_back(readFuncParam(in));
			}
		}

		requireToken(in, ")");
		if constexpr (In::settings() & sluSyn)
		{
			if (checkReadToken(in,"->"))
				ret.retType = readTypeExpr(in,false);
		}
		return ret;
	}
	//Pair{fn,hasError}
	template<AnyInput In>
	inline std::pair<std::variant<Function<In>, FunctionInfo<In>>,bool> readFuncBody(In& in,std::optional<std::string> funcName)
	{
		Position place = in.getLoc();
		if constexpr (In::settings() & sluSyn)
			in.genData.pushLocalScope();

		FunctionInfo<In> fi = readFuncInfo(in);
		if constexpr (In::settings()&sluSyn)
		{
			skipSpace(in);
			if (!in || (in.peek() != '{'))//no { found?
			{
				fi.local2Mp = in.genData.popLocalScope();
				return { std::move(fi), false };//No block, just the info
			}
		}
		Function<In> func = { std::move(fi) };

		if constexpr (In::settings() & sluSyn)
		{
			try {
				requireToken(in, "{");
				if(funcName.has_value())
					in.genData.pushScope(in.getLoc(),*funcName);
				func.block = readBlock<false>(in, func.hasVarArgParam, !funcName.has_value());
				requireToken(in, "}");
				func.local2Mp = in.genData.popLocalScope();
			} catch(const ParseError&)
			{
				func.local2Mp = in.genData.popLocalScope();
				throw;
			}
		}
		else
		{
			try
			{
				if (funcName.has_value())
					in.genData.pushScope(in.getLoc(), *funcName);
				func.block = readBlock<false>(in, func.hasVarArgParam, !funcName.has_value());
			}
			catch (const ParseError& e)
			{
				in.handleError(e.m);

				if (recoverErrorTextToken(in, "end"))
					return { std::move(func),true };// Found it, recovered!

				//End of stream, and no found end's, maybe the error is a missing "end"?
				throw FailedRecoveryError(std::format(
					"Missing " LUACC_SINGLE_STRING("end") ", maybe for " LC_function " at {} ?",
					errorLocStr(in, place)
				));
			}
			requireToken(in, "end");
		}
		return { std::move(func), false };
	}

	template<bool isLoop,SemicolMode semicolMode = SemicolMode::REQUIRE, AnyInput In>
	inline Block<In> readDoOrStatOrRet(In& in, const bool allowVarArg)
	{
		if constexpr(In::settings() & sluSyn)
		{
			skipSpace(in);
			if (in.peek() == '{')
			{
				in.skip();
				return readBlockNoStartCheck<isLoop>(in,allowVarArg,true);
			}

			in.genData.pushAnonScope(in.getLoc());//readBlock also pushes!

			if (readReturn<semicolMode>(in, allowVarArg))
				return in.genData.popScope(in.getLoc());
			//Basic Statement + ';'

			readStatement<isLoop>(in, allowVarArg);

			Block<In> bl = in.genData.popScope(in.getLoc());

			if constexpr (semicolMode == SemicolMode::REQUIRE_OR_KW)
			{
				skipSpace(in);

				const char ch1 = in.peek();

				if (ch1 == ';')
					in.skip();//thats it
				else if (!isBasicBlockEnding(in, ch1))
					throwSemicolMissingAfterStat(in);
			}
			else if constexpr(semicolMode==SemicolMode::NONE)
				readOptToken(in, ";");
			else
				requireToken(in, ";");

			return bl;
		}
		requireToken(in, "do");
		Block<In> bl = readBlock<isLoop>(in,allowVarArg, true);
		requireToken(in, "end");

		return bl;
	}

	template<bool isLoop,bool BASIC, AnyInput In>
	inline Soe<In> readSoe(In& in, const bool allowVarArg)
	{
		if (checkReadToken(in, "=>"))
		{
			return std::make_unique<Expression<In>>(readExpr<BASIC>(in, allowVarArg));
		}
		return readDoOrStatOrRet<isLoop, SemicolMode::NONE>(in, allowVarArg);
	}

	template<bool isLoop,AnyInput In>
	inline bool readUchStat(In& in, const Position place, const ExportData exported)
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
				case 'f':
					if (readFchStat<isLoop>(in, place, exported, OptSafety::UNSAFE, false))
						return true;
					break;
				case '{':
				{
					in.skip();
					in.genData.pushUnsafe();
					StatementType::UnsafeBlock<In> res = { readBlockNoStartCheck<isLoop>(in, false,true) };
					in.genData.popSafety();
					in.genData.addStat(place, std::move(res));
					return true;
				}
				case 'e':
					if (!exported && readEchStat<isLoop>(in, place, OptSafety::UNSAFE, false))
						return true;
					break;
				case 't'://traits?
				default:
					break;
				}
				throwExpectedUnsafeable(in);
			}
			break;
		case 'i':
			if (checkReadTextToken(in, "union"))
			{
				readStructStat<StatementType::Union<In>, true>(in, place, exported);
				return true;
			}
		default:
			break;
		}
		return false;
	}

	template<bool isLoop, AnyInput In>
	inline bool readEchStat(In& in, const Position place, const OptSafety safety, const bool allowVarArg)
	{
		if(checkReadTextToken(in,"extern"))
		{
			skipSpace(in);
			std::string abi = readStringLiteral(in, in.peek());
			Position abiEnd = in.getLoc();
			skipSpace(in);
			if (in.peek() == '{')
			{
				in.skip();
				StatementType::ExternBlock<In> res{};

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
	inline bool readFchStat(In& in, const Position place, const ExportData exported,const OptSafety safety, const bool allowVarArg)
	{
		if (in.isOob(1))
			return false;
		const char ch2 = in.peekAt(1);
		switch (ch2)
		{
		case 'n':
			if constexpr (In::settings() & sluSyn)
			{
				if (checkReadTextToken(in, "fn"))
				{
					readFunctionStatement<isLoop, StatementType::FN<In>,StatementType::FnDecl<In>>(
						in, place, allowVarArg, exported, safety
					);
					return true;
				}
			}
			break;
		case 'u':
			if (checkReadTextToken(in, "function"))
			{
				readFunctionStatement<isLoop, StatementType::FUNCTION_DEF<In>, StatementType::FunctionDecl<In>>(
					in, place, allowVarArg, exported, safety
				);
				return true;
			}
			break;
		case 'o':
			if constexpr (In::settings() & sluSyn)
			{
				if (exported || safety!=OptSafety::DEFAULT)
					break;
			}
			if (checkReadTextToken(in, "for"))
			{
				/*
				 for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end |
				 for namelist in explist do block end |
				*/

				Sel<In::settings()& sluSyn, NameList<In>, PatV<true,true>> names;
				if constexpr (In::settings() & sluSyn)
				{
					skipSpace(in);
					names = readPat<true>(in, true);
				}
				else
					names = readNameList(in);

				bool isNumeric = false;

				if constexpr (In::settings() & sluSyn)
					isNumeric = checkReadToken(in, "=");
				else//1 name, then MAYBE equal
					isNumeric = names.size() == 1 && checkReadToken(in, "=");
				if (isNumeric)
				{
					StatementType::FOR_LOOP_NUMERIC<In> res{};
					if constexpr (In::settings() & sluSyn)
						res.varName = std::move(names);
					else
						res.varName = names[0];

					// for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end | 
					res.start = readExpr(in, allowVarArg);
					requireToken(in, ",");
					res.end = readExpr<In::settings() & sluSyn>(in, allowVarArg);

					if (checkReadToken(in, ","))
						res.step = readExpr<In::settings() & sluSyn>(in, allowVarArg);



					res.bl = readDoOrStatOrRet<true>(in, allowVarArg);

					in.genData.addStat(place, std::move(res));
					return true;
				}
				// Generic Loop
				// for namelist in explist do block end | 

				StatementType::FOR_LOOP_GENERIC<In> res{};
				res.varNames = std::move(names);

				requireToken(in, "in");
				if constexpr (In::settings() & sluSyn)
					res.exprs = readExpr<true>(in, allowVarArg);
				else
					res.exprs = readExpList(in, allowVarArg);


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
	inline bool readLchStat(In& in, const Position place, const ExportData exported, const bool allowVarArg)
	{
		if (in.isOob(1))
			return false;

		const char ch2 = in.peekAt(1);
		switch (ch2)
		{
		case 'e':
			if constexpr (In::settings() & sluSyn)
			{
				if (checkReadTextToken(in, "let"))
				{
					readVarStatement<true, isLoop, StatementType::LET<In>>(in, place, allowVarArg, exported);
					return true;
				}
			}
			break;
		case 'o':
			if (checkReadTextToken(in, "local"))
			{
				/*
					local function Name funcbody |
					local attnamelist [‘=’ explist]
				*/
				if constexpr (!(In::settings() & sluSyn))
				{
					if (checkReadTextToken(in, "function"))
					{ // local function Name funcbody
						//NOTE: no real function decl, as `local function` is not in slu.
						readFunctionStatement<isLoop, StatementType::LOCAL_FUNCTION_DEF<In>,StatementType::FunctionDecl<In>>(
							in, place, allowVarArg, false,OptSafety::DEFAULT
						);
						return true;
					}
				}
				// Local Variable
				readVarStatement<true, isLoop, StatementType::LOCAL_ASSIGN<In>>(in, place, allowVarArg, exported);
				return true;
			}
			break;
		default:
			break;
		}
		return false;
	}

	template<bool isLoop, AnyInput In>
	inline bool readCchStat(In& in, const Position place, const ExportData exported, const bool allowVarArg)
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
				readVarStatement<false,isLoop, StatementType::CONST<In>>(in, place, allowVarArg, exported);
				return true;
			}
			break;
		default:
			break;
		}
		return false;
	}


	template<bool isLoop,AnyInput In>
	inline bool readSchStat(In& in, const Position place, const ExportData exported)
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
					if(!exported && readEchStat<isLoop>(in,place, OptSafety::SAFE, false))
						return true;
					break;
				case 'f':
					if (readFchStat<isLoop>(in, place, exported, OptSafety::SAFE, false))
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
				readStructStat<StatementType::Struct<In>,false>(in, place, exported);
				return true;
			}
			break;
		default:
			break;
		}
		return false;
	}

	template<bool isLoop,class StatT,class DeclStatT, AnyInput In>
	inline void readFunctionStatement(In& in, 
		const Position place, const bool allowVarArg, 
		const ExportData exported, const OptSafety safety)
	{
		StatT res{};
		std::string name;//moved @ readFuncBody
		if constexpr (In::settings() & sluSyn)
			name = readName(in);
		else
			name = readFuncName(in);
		res.name = in.genData.addLocalObj(name);
		res.place = in.getLoc();

		try
		{
			auto [fun, err] = readFuncBody(in,std::move(name));
			if(ezmatch(std::move(fun))(
				varcase(Function<In>&&) {
					res.func = std::move(var);
					return false;
				},
				varcase(FunctionInfo<In>&&) {
					DeclStatT declRes{ std::move(var) };
					declRes.name = res.name;
					declRes.place = res.place;

					if constexpr (In::settings() & sluSyn)
					{
						declRes.exported = exported;
						declRes.safety = safety;
					}

					in.genData.addStat(place, std::move(declRes));
					return true;
				}
			))
				return;//Staement was added

			if (err)
			{
				in.handleError(std::format(
					"In " LC_function " " LUACC_SINGLE_STRING("{}") " at {}",
					in.genData.asSv(res.name), errorLocStr(in, res.place)
				));
			}
		}
		catch (const ParseError& e)
		{
			in.handleError(e.m);
			throw ErrorWhileContext(std::format(
				"In " LC_function " " LUACC_SINGLE_STRING("{}") " at {}",
				in.genData.asSv(res.name), errorLocStr(in, res.place)
			));
		}

		if constexpr (In::settings() & sluSyn)
		{
			res.exported = exported;
			res.func.safety = safety;
		}

		return in.genData.addStat(place, std::move(res));
	}
	//TODO: handle basic (in basic expressions, if expressions can only have basic expressions)
	template<bool isLoop,bool forExpr,bool BASIC, AnyInput In>
	inline auto readIfCond(In& in, const bool allowVarArg)
	{
		BaseIfCond<In,forExpr> res{};

		res.cond = mayBoxFrom<forExpr>(readBasicExpr(in, allowVarArg));

		if constexpr (In::settings() & sluSyn)
		{
			res.bl = mayBoxFrom<forExpr>(readSoe<isLoop, false>(in, allowVarArg));

			while (checkReadTextToken(in, "else"))
			{
				if (checkReadTextToken(in, "if"))
				{
					Expression<In> elExpr = readBasicExpr(in, allowVarArg);
					Soe<In> elBlock = readSoe<isLoop, false>(in, allowVarArg);

					res.elseIfs.emplace_back(std::move(elExpr), std::move(elBlock));
					continue;
				}

				res.elseBlock = mayBoxFrom<forExpr>(readSoe<isLoop, false>(in, allowVarArg));
				break;
			}
		}
		else
		{
			requireToken(in, "then");
			res.bl = wontBox(readBlock<isLoop>(in, allowVarArg,true));
			while (checkReadTextToken(in, "elseif"))
			{
				Expression<In> elExpr = readExpr(in, allowVarArg);
				requireToken(in, "then");
				Block<In> elBlock = readBlock<isLoop>(in, allowVarArg, true);

				res.elseIfs.emplace_back(std::move(elExpr), std::move(elBlock));
			}

			if (checkReadTextToken(in, "else"))
				res.elseBlock = wontBox(readBlock<isLoop>(in, allowVarArg, true));

			requireToken(in, "end");
		}
		return res;
	}
	template<bool isLocal, bool isLoop, class StatT, AnyInput In >
	inline void readVarStatement(In& in, const Position place, const bool allowVarArg, const ExportData exported)
	{
		StatT res;
		if constexpr (In::settings() & sluSyn)
		{
			skipSpace(in);
			res.names = readPat<isLocal>(in, true);
			res.exported = exported;
			if constexpr (!isLocal)
				in.genData.pushLocalScope();
		}
		else
			res.names = readAttNameList(in);

		if (checkReadToken(in, "="))
		{// [‘=’ explist]
			res.exprs = readExpList(in, allowVarArg);
		}
		if constexpr (!isLocal)
			res.local2Mp = in.genData.popLocalScope();
		return in.genData.addStat(place, std::move(res));
	}

	template<bool isLoop, AnyInput In>
	inline void readStatement(In& in,const bool allowVarArg)
	{
		/*
		 varlist ‘=’ explist |
		 functioncall |
		*/

		skipSpace(in);

		const Position place = in.getLoc();
		Statement<In> ret;

		const char firstChar = in.peek();
		switch (firstChar)
		{
		case ';':
			in.skip();
			return in.genData.addStat(place, StatementType::SEMICOLON{});
		case ':'://may be label
			if constexpr (In::settings() & sluSyn)
			{
				if(in.peekAt(1) == '>')
					break;//Not a label
			}
			return readLabel(in, place);

		case 'f'://for?, function?, fn?
			if(readFchStat<isLoop>(in, place, false,OptSafety::DEFAULT, allowVarArg))
				return;
			break;
		case 'l'://local?
			if (readLchStat<isLoop>(in, place, false, allowVarArg))
				return;

			break;
		case 'c'://const comptime?
			if constexpr (In::settings() & sluSyn)
			{
				if(readCchStat<isLoop>(in, place, false, allowVarArg))
					return;
			}
			break;
		case '{':// ‘{’ block ‘}’
			if constexpr (In::settings() & sluSyn)
			{
				in.skip();//Skip ‘{’
				return in.genData.addStat(place,
					StatementType::BLOCK<In>(readBlockNoStartCheck<isLoop>(in, allowVarArg,true))
				);
			}
			break;
		case 'd'://do?
			if constexpr (In::settings() & sluSyn)
			{
				if (checkReadTextToken(in, "drop"))
				{
					return in.genData.addStat(place,
						StatementType::DROP<In>(readExpr(in,allowVarArg))
					);
				}
			}
			else
			{
				if (checkReadTextToken(in, "do")) // ‘do’ block ‘end’
				{
					return in.genData.addStat(place,
						StatementType::BLOCK<In>(readBlockNoStartCheck<isLoop>(in, allowVarArg,true))
					);
				}
			}
			break;
		case 'b'://break?
			if (checkReadTextToken(in, "break"))
			{
				if constexpr (!isLoop)
				{
					in.handleError(std::format(
						"Break used outside of loop{}"
						, errorLocStr(in)));
				}
				return in.genData.addStat(place, StatementType::BREAK{});
			}
			break;
		case 'g'://goto?
			if (checkReadTextToken(in, "goto"))//goto Name
			{
				return in.genData.addStat(place,
					StatementType::GOTO<In>(in.genData.resolveName(readName(in)))
				);
			}
			break;
		case 'w'://while?
			if (checkReadTextToken(in, "while"))
			{ // while exp do block end

				Expression<In> expr = readBasicExpr(in,allowVarArg);

				Block<In> bl = readDoOrStatOrRet<true>(in,allowVarArg);
				return in.genData.addStat(place, 
					StatementType::WHILE_LOOP<In>(std::move(expr), std::move(bl))
				);
			}
			break;
		case 'r'://repeat?
			if (checkReadTextToken(in, "repeat"))
			{ // repeat block until exp
				Block<In> bl;
				if constexpr (In::settings() & sluSyn)
					bl = readDoOrStatOrRet<true, SemicolMode::NONE>(in, allowVarArg);
				else
					bl = readBlock<true>(in, allowVarArg, true);
				requireToken(in, "until");
				Expression<In> expr = readExpr(in,allowVarArg);

				return in.genData.addStat(place, 
					StatementType::REPEAT_UNTIL<In>({ std::move(expr), std::move(bl) })
				);
			}
			break;
		case 'i'://if?
			if (checkReadTextToken(in, "if"))
			{ // if exp then block {elseif exp then block} [else block] end
				return in.genData.addStat(place, 
					readIfCond<isLoop,false,false>(
						in,allowVarArg
				));
			}
			break;

			//Slu
		case 'e'://ex ...?
			if constexpr (In::settings() & sluSyn)
			{
				if (checkReadTextToken(in, "ex"))
				{
					skipSpace(in);
					switch (in.peek())
					{
					case 'f'://fn? function?
						if (readFchStat<isLoop>(in, place, true, OptSafety::DEFAULT, allowVarArg))
							return;
						break;
					case 't'://trait?
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
				else if (readEchStat<isLoop>(in, place, OptSafety::DEFAULT, allowVarArg))
					return;
			}
			break;
		case 's'://safe? struct?
			if constexpr (In::settings() & sluSyn)
			{
				if(readSchStat<isLoop>(in, place,false))
					return;
			}
			break;
		case 'u'://use? unsafe?
			if constexpr (In::settings() & sluSyn)
			{
				if (readUchStat<isLoop>(in, place, false))
					return;
			}
			break;
		case 'm'://mod?
			if constexpr (In::settings() & sluSyn)
			{
				if (readModStat(in, place, false))
					return;
			}
			break;
		case 't'://trait?
			if constexpr (In::settings() & sluSyn)
			{
			}
			break;
		default://none of the above...
			break;
		}

		in.genData.addStat(place, 
			parsePrefixExprVar<StatementData<In>,false>(
				in,allowVarArg, firstChar
		));
	}

	/**
	 * @throws slu::parse::ParseFailError
	 */
	template<AnyInput In>
	inline ParsedFile<In> parseFile(In& in)
	{
		if constexpr (!(In::settings() & sluSyn))
			in.genData.pushLocalScope();
		try
		{
			ParsedFile<In> res;
			if constexpr (In::settings() & sluSyn)
			{
				auto [sl, mp] = readStatList<false>(in, true, true);
				res.code = std::move(sl);
				res.mp = mp;
			}
			else
				res.code = readBlock<false>(in, true, true);

			_ASSERT(in.genData.scopes.empty());

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
			if constexpr (!(In::settings() & sluSyn))
				res.local2Mp = in.genData.popLocalScope();
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