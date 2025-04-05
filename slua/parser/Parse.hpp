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

#include "basic/ReadMiscNames.hpp"
#include "adv/ReadName.hpp"
#include "adv/RequireToken.hpp"
#include "adv/SkipSpace.hpp"
#include "adv/ReadExpr.hpp"
#include "adv/ReadStringLiteral.hpp"
#include "adv/ReadTable.hpp"
#include "adv/RecoverFromError.hpp"
#include "adv/ReadType.hpp"



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


namespace sluaParse
{
	//Doesnt skip space, the current character must be a valid args starter
	template<AnyInput In>
	inline Args<In> readArgs(In& in, const bool allowVarArg)
	{
		const char ch = in.peek();
		if (ch == '"' || ch=='\'' || ch=='[')
		{
			return ArgsType::LITERAL(readStringLiteral(in, ch));
		}
		else if (ch == '(')
		{
			in.skip();//skip start
			skipSpace(in);
			ArgsType::EXPLIST<In> res{};
			if (in.peek() == ')')// Check if 0 args
			{
				in.skip();
				return res;
			}
			res.v = readExpList(in,allowVarArg);
			requireToken(in, ")");
			return res;
		}
		else if (ch == '{')
		{
			return ArgsType::TABLE<In>(readTableConstructor(in,allowVarArg));
		}
		throw UnexpectedCharacterError(std::format(
			"Expected function arguments ("
			LUACC_SINGLE_STRING(",")
			" or "
			LUACC_SINGLE_STRING(";")
			"), found " LUACC_SINGLE_STRING("{}")
			"{}"
		, ch, errorLocStr(in)));
	}

	//startCh == in.peek() !!!
	inline bool isBasicBlockEnding(AnyInput auto& in, const char startCh)
	{
		if constexpr (in.settings() & sluaSyn)
		{
			if (startCh == '}') return true;
			if (startCh == 'u')
			{
				if (checkTextToken(in, "until"))
					return true;
			}
			else if (startCh == 'e')
			{
				if (checkTextToken(in, "else"))
					return true;

			}
			return false;
		}

		if (startCh == 'u')
		{
			if (checkTextToken(in, "until"))
				return true;
		}
		else if (startCh == 'e')
		{
			const char ch1 = in.peekAt(1);
			if (ch1 == 'n')
			{
				if (checkTextToken(in, "end"))
					return true;
			}
			else if (ch1 == 'l')
			{
				if (checkTextToken(in, "else") || checkTextToken(in, "elseif"))
					return true;
			}
		}
		return false;
	}
	enum class SemicolMode
	{
		NONE,REQUIRE,REQUIRE_OR_KW
	};

	template<SemicolMode semicolReq, AnyInput In>
	inline bool readReturn(In& in, const bool allowVarArg, Block<In>& ret)
	{
		if (checkReadTextToken(in, "return"))
		{
			ret.hadReturn = true;

			skipSpace(in);

			const char ch1 = in.peek();

			if (ch1 == ';')
				in.skip();//thats it
			else if (isBasicBlockEnding(in, ch1))
			{
				if constexpr (semicolReq==SemicolMode::REQUIRE)
					requireToken(in, ";");//Lazy way to throw error, maybe fix later?
			}
			else
			{
				ret.retExprs = readExpList(in, allowVarArg);

				if constexpr (semicolReq!=SemicolMode::NONE)
					requireToken(in, ";");
				else
					readOptToken(in, ";");
			}
			return true;
		}
		return false;
	}

	template<bool isLoop,AnyInput In>
	inline Block<In> readBlock(In& in,const bool allowVarArg)
	{
		/*
			block ::= {stat} [retstat]
			retstat ::= return [explist] [‘;’]
		*/

		skipSpace(in);

		Block<In> ret{};
		ret.start = in.getLoc();

		while (true)
		{
			skipSpace(in);

			if (!in)//File ended, so block ended too
				break;

			const char ch = in.peek();

			if (ch == 'r')
			{
				if (readReturn<SemicolMode::NONE>(in, allowVarArg, ret))
					break;// no more loop
			}
			else if (isBasicBlockEnding(in, ch))
				break;// no more loop

			// Not some end / return keyword, must be a statement

			ret.statList.emplace_back(readStatment<isLoop>(in, allowVarArg));
		}
		ret.end = in.getLoc();
		return ret;
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
			{
				Parameter<In> p;
				p.name = readName(in);

				ret.params.emplace_back(std::move(p));
			}

			while (checkReadToken(in, ","))
			{
				if (checkReadToken(in, "..."))
				{
					ret.hasVarArgParam = true;
					break;//cant have anything after the ... arg
				}
				Parameter<In> p;
				p.name = readName(in);

				ret.params.emplace_back(std::move(p));
			}
		}

		requireToken(in, ")");
		if constexpr(in.settings()&sluaSyn)
			requireToken(in, "{");
		try
		{
			ret.block = readBlock<false>(in, ret.hasVarArgParam);
		}
		catch (const ParseError& e)
		{
			in.handleError(e.m);

			if(recoverErrorTextToken(in,"end"))
				return { std::move(ret),true };// Found it, recovered!

			//End of stream, and no found end's, maybe the error is a missing "end"?
			throw FailedRecoveryError(std::format(
				"Missing " LUACC_SINGLE_STRING("end") ", maybe for " LC_function " at {} ?",
				errorLocStr(in, place)
			));
		}
		requireToken(in, sel<In>("end","}"));

		return { std::move(ret),false };
	}

	template<AnyInput In>
	inline std::string readLabel(In& in)
	{
		//label ::= ‘::’ Name ‘::’
		//SL label ::= ‘:::’ Name ‘:’

		requireToken(in, sel<In>("::", ":::"));

		const std::string res = readName(in);

		requireToken(in, sel<In>("::", ":"));

		return res;
	}

	template<AnyInput In>
	inline bool readExportableStat(In& in,StatementData<In>& outData,bool exported)
	{
		if (checkReadTextToken(in, "type"))
		{
			StatementType::TYPE res{};
			res.exported = exported;

			res.name = readName(in);
			requireToken(in, "=");
			res.ty = readType(in);

			outData = std::move(res);
			return true;
		}
		return false;
	}

	template<bool isLoop, AnyInput In>
	inline Block<In> readBlockNoStartCheck(In& in, const bool allowVarArg)
	{
		Block<In> bl = readBlock<isLoop>(in, allowVarArg);
		requireToken(in, sel<In>("end","}"));

		return (bl);
	}

	template<bool isLoop,SemicolMode semicolMode = SemicolMode::REQUIRE, AnyInput In>
	inline Block<In> readDoOrStatOrRet(In& in, const bool allowVarArg)
	{
		if constexpr(in.settings() & sluaSyn)
		{
			skipSpace(in);
			if (in.peek() == '{')
			{
				in.skip();
				return readBlockNoStartCheck<isLoop>(in,allowVarArg);
			}
			Block<In> bl{};
			bl.start = in.getLoc();

			if (readReturn<semicolMode>(in, allowVarArg, bl))
			{
				bl.end = in.getLoc();
				return bl;
			}
			//Basic Statement + ';'

			bl.statList.push_back(readStatment<isLoop>(in, allowVarArg));

			bl.end = in.getLoc();


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

	template<bool isLoop, AnyInput In>
	inline Statement<In> readStatment(In& in,const bool allowVarArg)
	{
		/*
		 varlist ‘=’ explist |
		 functioncall |
		*/

		skipSpace(in);

		Statement<In> ret;
		ret.place = in.getLoc();

		const char firstChar = in.peek();
		switch (firstChar)
		{
		case ';':
			in.skip();
			ret.data = StatementType::SEMICOLON();
			return ret;

		case ':'://must be label
			ret.data = StatementType::LABEL(readLabel(in));
			return ret;

		case 'f'://for?, function?, fn?
			if (checkReadTextToken(in, "for"))
			{
				/*
				 for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end |
				 for namelist in explist do block end |
				*/

				if constexpr (in.settings() & sluaSyn) requireToken(in, "(");

				NameList names = readNameList(in);

				if (names.size() == 1 && checkReadToken(in, "="))//1 name, then MAYBE equal
				{
					StatementType::FOR_LOOP_NUMERIC<In> res{};
					res.varName = names[0];

					// for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end | 
					res.start = readExpr(in,allowVarArg);
					requireToken(in, ",");
					res.end = readExpr(in,allowVarArg);

					if (checkReadToken(in, ","))
						res.step = readExpr(in,allowVarArg);

					if constexpr (in.settings() & sluaSyn) requireToken(in, ")");

					res.bl = readDoOrStatOrRet<true>(in,allowVarArg);

					ret.data = std::move(res);
					return ret;
				}
				// Generic Loop
				// for namelist in explist do block end | 

				StatementType::FOR_LOOP_GENERIC<In> res{};
				res.varNames = names;

				requireToken(in, "in");
				res.exprs = readExpList(in,allowVarArg);

				if constexpr (in.settings() & sluaSyn) requireToken(in, ")");

				res.bl = readDoOrStatOrRet<true>(in,allowVarArg);

				ret.data = std::move(res);
				return ret;
			}
			if (checkReadTextToken(in, "function"))
			{ // function funcname funcbody
				StatementType::FUNCTION_DEF<In> res{};

				res.place = in.getLoc();

				res.name = readFuncName(in);

				try
				{
					auto [fun, err] = readFuncBody(in);
					res.func = std::move(fun);
					if(err)
					{
						in.handleError(std::format(
							"In " LC_function " " LUACC_SINGLE_STRING("{}") " at {}",
							res.name, errorLocStr(in, res.place)
						));
					}
				}
				catch (const ParseError& e)
				{
					in.handleError(e.m);
					throw ErrorWhileContext(std::format(
						"In " LC_function " " LUACC_SINGLE_STRING("{}") " at {}",
						res.name, errorLocStr(in, res.place)
					));
				}

				ret.data = std::move(res);
				return ret;
			}
			break;
		case 'l'://local?
			if (checkReadTextToken(in, "local"))
			{//func, or var
				/*
					local function Name funcbody |
					local attnamelist [‘=’ explist]
				*/
				if (checkReadTextToken(in, "function"))
				{ // local function Name funcbody
					StatementType::LOCAL_FUNCTION_DEF<In> res;

					res.place = in.getLoc();

					res.name = readFuncName(in);

					try
					{
						auto [fun, err] = readFuncBody(in);
						res.func = std::move(fun);
						if (err)
						{

							in.handleError(std::format(
								"In " LC_function " " LUACC_SINGLE_STRING("{}") " at {}",
								res.name, errorLocStr(in, res.place)
							));
						}
					}
					catch (const ParseError& e)
					{
						in.handleError(e.m);
						throw ErrorWhileContext(std::format(
							"In " LC_function " " LUACC_SINGLE_STRING("{}") " at {}",
							res.name, errorLocStr(in, res.place)
						));
					}

					ret.data = std::move(res);
					return ret;
				}
				// Local Variable

				StatementType::LOCAL_ASSIGN<In> res;
				if constexpr(in.settings() & sluaSyn)
					res.names = readNameList(in);
				else
					res.names = readAttNameList(in);

				if (checkReadToken(in, "="))
				{// [‘=’ explist]
					res.exprs = readExpList(in,allowVarArg);
				}
				ret.data = std::move(res);
				return ret;
			}
			break;
		case '{':// ‘{’ block ‘}’
			if constexpr (in.settings() & sluaSyn)
			{
				in.skip();//Skip ‘}’
				ret.data = StatementType::BLOCK<In>(readBlockNoStartCheck<isLoop>(in, allowVarArg));
				return ret;
			}
			break;
		case 'd'://do?
			if constexpr (!(in.settings() & sluaSyn))
			{
				if (checkReadTextToken(in, "do")) // ‘do’ block ‘end’
				{
					ret.data = StatementType::BLOCK<In>(readBlockNoStartCheck<isLoop>(in, allowVarArg));
					return ret;
				}
			}
			break;
		case 'b'://break?
			if (checkReadTextToken(in, "break"))
			{
				if constexpr (!isLoop)
				{
					throw ReservedNameError(std::format(
						"Break used outside of loop{}"
						, errorLocStr(in)));
				}
				ret.data = StatementType::BREAK();
				return ret;
			}
			break;
		case 'g'://goto?
			if (checkReadTextToken(in, "goto"))//goto Name
			{
				ret.data = StatementType::GOTO(readName(in));
				return ret;
			}
			break;
		case 'w'://while?
			if (checkReadTextToken(in, "while"))
			{ // while exp do block end

				Expression<In> expr = readExprParens(in,allowVarArg);

				Block<In> bl = readDoOrStatOrRet<true>(in,allowVarArg);
				ret.data = StatementType::WHILE_LOOP<In>(std::move(expr), std::move(bl));
				return ret;
			}
			break;
		case 'r'://repeat?
			if (checkReadTextToken(in, "repeat"))
			{ // repeat block until exp
				Block<In> bl;
				if constexpr (in.settings() & sluaSyn)
					bl = readDoOrStatOrRet<true, SemicolMode::NONE>(in, allowVarArg);
				else
					bl = readBlock<true>(in, allowVarArg);
				requireToken(in, "until");
				Expression<In> expr = readExpr(in,allowVarArg);

				ret.data = StatementType::REPEAT_UNTIL<In>({ std::move(expr), std::move(bl) });
				return ret;
			}
			break;
		case 'i'://if?
			if (checkReadTextToken(in, "if"))
			{ // if exp then block {elseif exp then block} [else block] end

				StatementType::IF_THEN_ELSE<In> res{};

				res.cond = readExprParens(in,allowVarArg);

				if constexpr (in.settings() & sluaSyn)
				{
					res.bl = readDoOrStatOrRet<isLoop, SemicolMode::REQUIRE_OR_KW>(in, allowVarArg);

					while (checkReadTextToken(in, "else"))
					{
						if (checkReadTextToken(in, "if"))
						{
							Expression<In> elExpr = readExprParens(in, allowVarArg);
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


				ret.data = std::move(res);
				return ret;
			}
			break;

			//Slua
		case 'e'://ex ...?
			if constexpr (in.settings() & sluaSyn)
			{
				if (checkReadTextToken(in, "ex"))
				{
					if (readExportableStat(in,ret.data,true))
						return ret;
				}
			}
			break;
		case 't'://type?
			if constexpr (in.settings() & sluaSyn)
			{
				if (readExportableStat(in, ret.data, false))
					return ret;
			}
			break;
		default://none of the above...
			break;
		}

		ret.data = parsePrefixExprVar<StatementData<In>,false>(in,allowVarArg, firstChar);
		return ret;
	}

	template<bool isSlua>
	struct ParsedFileV
	{
		//TypeList types
		BlockV<isSlua> code;
	};


	template<AnyCfgable CfgT>
	using ParsedFile = SelV<CfgT, ParsedFileV>;

	/**
	 * @throws sluaParse::ParseFailError
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