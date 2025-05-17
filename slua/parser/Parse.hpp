/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/Input.hpp>
#include <slua/parser/State.hpp>

#include "basic/ReadArgs.hpp"
#include "basic/ReadMiscNames.hpp"
#include "basic/ReadBasicStats.hpp"
#include "basic/ReadModStat.hpp"
#include "basic/ReadUseStat.hpp"
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

	TODO:

	[_] EOS handling

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


namespace slua::parse
{
	template<AnyInput In>
	inline Parameter<In> readFuncParam(In& in)
	{
		Parameter<In> p;
		if constexpr (In::settings() & sluaSyn)
			p.name = readPat(in, true);
		else
			p.name = in.genData.resolveUnknown(readName(in));
		
		return p;
	}
	//Pair{fn, hadErr}
	template<AnyInput In>
	inline std::pair<Function<In>,bool> readFuncBody(In& in)
	{
		/*
			funcbody ::= ‘(’ [parlist] ‘)’ block end
			parlist ::= namelist [‘,’ ‘...’] | ‘...’
		*/
		Function<In> ret{};

		Position place = in.getLoc();

		requireToken(in, "(");

		skipSpace(in);

		const char ch = in.peek();

		if (ch == '.')
		{
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
					ret.hasVarArgParam = true;
					break;//cant have anything after the ... arg
				}
				ret.params.emplace_back(readFuncParam(in));
			}
		}

		requireToken(in, ")");
		if constexpr (In::settings() & sluaSyn)
		{
			if (checkReadToken(in,"->"))
				ret.retType = readTypeExpr(in,false);

			ret.block = readDoOrStatOrRet<false>(in, ret.hasVarArgParam);
		}
		else
		{
			try
			{
				ret.block = readBlock<false>(in, ret.hasVarArgParam);
			}
			catch (const ParseError& e)
			{
				in.handleError(e.m);

				if (recoverErrorTextToken(in, "end"))
					return { std::move(ret),true };// Found it, recovered!

				//End of stream, and no found end's, maybe the error is a missing "end"?
				throw FailedRecoveryError(std::format(
					"Missing " LUACC_SINGLE_STRING("end") ", maybe for " LC_function " at {} ?",
					errorLocStr(in, place)
				));
			}
			requireToken(in, "end");
		}

		return { std::move(ret),false };
	}

	template<bool isLoop,SemicolMode semicolMode = SemicolMode::REQUIRE, AnyInput In>
	inline Block<In> readDoOrStatOrRet(In& in, const bool allowVarArg)
	{
		if constexpr(In::settings() & sluaSyn)
		{
			skipSpace(in);
			if (in.peek() == '{')
			{
				in.skip();
				return readBlockNoStartCheck<isLoop>(in,allowVarArg);
			}

			in.genData.pushAnonScope(in.getLoc());

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
		Block<In> bl = readBlock<isLoop>(in,allowVarArg);
		requireToken(in, "end");

		return bl;
	}


	template<AnyInput In>
	inline bool readUchStat(In& in, const Position place, const ExportData exported)
	{
		const char ch2 = in.peekAt(1);
		switch (ch2)
		{
		case 's':
			if (readUseStat(in, place, exported))
				return true;
			break;
		case 'n':
			if (checkReadTextToken(in, "unsafe"))
			{
				throwExpectedUnsafeable(in);
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
			if constexpr (In::settings() & sluaSyn)
			{
				if (checkReadTextToken(in, "let"))
				{
					readVarStatement<isLoop, StatementType::LET<In>>(in, place, allowVarArg, exported);
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
				if constexpr (!(In::settings() & sluaSyn))
				{
					if (checkReadTextToken(in, "function"))
					{ // local function Name funcbody
						StatementType::LOCAL_FUNCTION_DEF<In> res;

						res.place = in.getLoc();

						res.name = in.genData.resolveUnknown(readFuncName(in));

						try
						{
							auto [fun, err] = readFuncBody(in);
							res.func = std::move(fun);
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

						in.genData.addStat(place, std::move(res));
						return true;
					}
				}
				// Local Variable
				readVarStatement<isLoop, StatementType::LOCAL_ASSIGN<In>>(in, place, allowVarArg, exported);
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
				readVarStatement<isLoop, StatementType::CONST<In>>(in, place, allowVarArg, exported);
				return true;
			}
			break;
		default:
			break;
		}
		return false;
	}


	template<AnyInput In>
	inline bool readSchStat(In& in, const Position place, const ExportData exported)
	{
		if (checkReadTextToken(in, "safe"))
		{
			//TODO
			throwExpectedSafeable(in);
		}
		return false;
	}

	template<bool isLoop,class StatT, AnyInput In>
	inline void readVarStatement(In& in, const Position place, const bool allowVarArg, const ExportData exported)
	{
		StatT res;
		if constexpr (In::settings() & sluaSyn)
		{
			res.names = readPat(in, true);
			res.exported = exported;
		}
		else
			res.names = readAttNameList(in);

		if (checkReadToken(in, "="))
		{// [‘=’ explist]
			res.exprs = readExpList(in, allowVarArg);
		}
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
		case ':'://must be label
			return readLabel(in, place);

		case 'f'://for?, function?, fn?
			if (checkReadTextToken(in, "for"))
			{
				/*
				 for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end |
				 for namelist in explist do block end |
				*/

				Sel<In::settings()&sluaSyn, NameList<In>,Pat> names;
				if constexpr (In::settings() & sluaSyn)
					names = readPat(in, true);
				else
					names = readNameList(in);
				
				bool isNumeric = false;

				if constexpr (In::settings() & sluaSyn)
					isNumeric = checkReadToken(in, "=");
				else//1 name, then MAYBE equal
					isNumeric = names.size() == 1 && checkReadToken(in, "=");
				if (isNumeric)
				{
					StatementType::FOR_LOOP_NUMERIC<In> res{};
					if constexpr (In::settings() & sluaSyn)
						res.varName = std::move(names);
					else
						res.varName = names[0];

					// for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end | 
					res.start = readExpr(in, allowVarArg);
					requireToken(in, ",");
					res.end = readExpr<In::settings() & sluaSyn>(in, allowVarArg);

					if (checkReadToken(in, ","))
						res.step = readExpr<In::settings() & sluaSyn>(in, allowVarArg);



					res.bl = readDoOrStatOrRet<true>(in, allowVarArg);

					return in.genData.addStat(place, std::move(res));
				}
				// Generic Loop
				// for namelist in explist do block end | 

				StatementType::FOR_LOOP_GENERIC<In> res{};
				res.varNames = std::move(names);

				requireToken(in, "in");
				if constexpr (In::settings() & sluaSyn)
					res.exprs = readExpr<true>(in, allowVarArg);
				else
					res.exprs = readExpList(in, allowVarArg);


				res.bl = readDoOrStatOrRet<true>(in,allowVarArg);

				return in.genData.addStat(place, std::move(res));
			}
			if (checkReadTextToken(in, "function"))
			{ // function funcname funcbody
				StatementType::FUNCTION_DEF<In> res{};

				res.place = in.getLoc();

				res.name = in.genData.resolveUnknown(readFuncName(in));

				try
				{
					auto [fun, err] = readFuncBody(in);
					res.func = std::move(fun);
					if(err)
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

				return in.genData.addStat(place, std::move(res));
			}
			break;
		case 'l'://local?
			if (readLchStat<isLoop>(in, place, false, allowVarArg))
				return;

			break;
		case 'c'://const comptime?
			if constexpr (In::settings() & sluaSyn)
			{
				if(readCchStat<isLoop>(in, place, false, allowVarArg))
					return;
			}
			break;
		case '{':// ‘{’ block ‘}’
			if constexpr (In::settings() & sluaSyn)
			{
				in.skip();//Skip ‘{’
				return in.genData.addStat(place,
					StatementType::BLOCK<In>(readBlockNoStartCheck<isLoop>(in, allowVarArg))
				);
			}
			break;
		case 'd'://do?
			if constexpr (In::settings() & sluaSyn)
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
						StatementType::BLOCK<In>(readBlockNoStartCheck<isLoop>(in, allowVarArg))
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
				if constexpr (In::settings() & sluaSyn)
					bl = readDoOrStatOrRet<true, SemicolMode::NONE>(in, allowVarArg);
				else
					bl = readBlock<true>(in, allowVarArg);
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

				StatementType::IF_THEN_ELSE<In> res{};

				res.cond = readBasicExpr(in,allowVarArg);

				if constexpr (In::settings() & sluaSyn)
				{
					res.bl = readDoOrStatOrRet<isLoop, SemicolMode::REQUIRE_OR_KW>(in, allowVarArg);

					while (checkReadTextToken(in, "else"))
					{
						if (checkReadTextToken(in, "if"))
						{
							Expression<In> elExpr = readBasicExpr(in, allowVarArg);
							Block<In> elBlock = readDoOrStatOrRet<isLoop, SemicolMode::REQUIRE_OR_KW>(in, allowVarArg);

							res.elseIfs.emplace_back(std::move(elExpr), std::move(elBlock));
							continue;
						}

						res.elseBlock = readDoOrStatOrRet<isLoop>(in, allowVarArg);
						break;
					}
				}
				else
				{
					requireToken(in, "then");
					res.bl = readBlock<isLoop>(in, allowVarArg);
					while (checkReadTextToken(in, "elseif"))
					{
						Expression<In> elExpr = readExpr(in, allowVarArg);
						requireToken(in, "then");
						Block<In> elBlock = readBlock<isLoop>(in, allowVarArg);

						res.elseIfs.emplace_back(std::move(elExpr), std::move(elBlock));
					}

					if (checkReadTextToken(in, "else"))
						res.elseBlock = readBlock<isLoop>(in, allowVarArg);

					requireToken(in, "end");
				}

				return in.genData.addStat(place, std::move(res));
			}
			break;

			//Slua
		case 'e'://ex ...?
			if constexpr (In::settings() & sluaSyn)
			{
				if (checkReadTextToken(in, "ex"))
				{
					skipSpace(in);
					switch (in.peek())
					{
					case 'f'://fn? function?
						break;
					case 't'://type?
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
						if (readUchStat(in, place, true))
							return;
						break;
					case 's'://safe?
						if (readSchStat(in, place, true))
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
			}
			break;
		case 's'://safe?
			if constexpr (In::settings() & sluaSyn)
			{
				if(readSchStat(in, place,false))
					return;
			}
			break;
		case 'u'://use? unsafe?
			if constexpr (In::settings() & sluaSyn)
			{
				if (readUchStat(in, place, false))
					return;
			}
			break;
		case 'm'://mod?
			if constexpr (In::settings() & sluaSyn)
			{
				if (readModStat(in, place, false))
					return;
			}
			break;
		case 't'://type?
			if constexpr (In::settings() & sluaSyn)
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
	 * @throws slua::parse::ParseFailError
	 */
	template<AnyInput In>
	inline ParsedFile<In> parseFile(In& in)
	{
		try
		{
			Block<In> bl = readBlock<false>(in, true);

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
			return { std::move(bl) };
		}
		catch (const BasicParseError& e)
		{
			in.handleError(e.m);
			throw ParseFailError();
		}
	}
}