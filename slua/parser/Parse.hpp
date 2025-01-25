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



/*

	TODO:

	[X] chunk ::= block

	[X] block ::= {stat} [retstat]

	[_] stat ::= [X] ‘;’ |
		[_] varlist ‘=’ explist |
		[_] functioncall |
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

	[_] varlist ::= var {‘,’ var}

	[_] var ::=  Name | prefixexp ‘[’ exp ‘]’ | prefixexp ‘.’ Name

	[X] namelist ::= Name {‘,’ Name}

	[X] explist ::= exp {‘,’ exp}

	[_] exp ::=  [X]nil | [X]false | [X]true | Numeral | [X]LiteralString | [X]‘...’ | [X]functiondef |
		 prefixexp | [X]tableconstructor | [X]exp binop exp | [X]unop exp

	[_] prefixexp ::= var | functioncall | ‘(’ exp ‘)’

	[_] functioncall ::=  prefixexp args | prefixexp ‘:’ Name args

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
	inline Args readArgs(AnyInput auto& in)
	{
		skipSpace(in);
		const char ch = in.peek();
		if (ch == '"' || ch=='\'' || ch=='[')
		{
			return ArgsType::LITERAL(readStringLiteral(in, ch));
		}
		else if (ch == '(')
		{
			skipSpace(in);
			ArgsType::EXPLIST res{};
			if (in.peek() == ')')// Check if 0 args
			{
				in.skip();
				return res;
			}
			res.v = readExpList(in);
			requireToken(in, ")");
			return res;
		}
		else if (ch == '{')
		{
			return ArgsType::TABLE(readTableConstructor(in));
		}
		throw UnexpectedCharacterError(
			"Expected function arguments ("
			LUACC_SINGLE_STRING(",")
			" or "
			LUACC_SINGLE_STRING(";")
			"), found " LUACC_START_SINGLE_STRING + ch + LUACC_END_SINGLE_STRING
			+ errorLocStr(in));
	}
	inline Var readVar(AnyInput auto& in)
	{
		/*
			var ::=  Name | prefixexp ‘[’ exp ‘]’ | prefixexp ‘.’ Name
			prefixexp ::= var | functioncall | ‘(’ exp ‘)’
			functioncall ::=  prefixexp args | prefixexp ‘:’ Name args

			--->

			var ::= baseVar {subvar}
			
			baseVar ::= Name | ‘(’ exp ‘)’ subvar

			funcArgs ::=  [‘:’ Name] args
			subvar ::= [funcArgs] ‘[’ exp ‘]’ | [funcArgs] ‘.’ Name
		*/
		return {};
	}


	//startCh == in.peek() !!!
	inline bool isBasicBlockEnding(AnyInput auto& in, const char startCh)
	{
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

	inline Block readBlock(AnyInput auto& in)
	{
		/*
			block ::= {stat} [retstat]
			retstat ::= return [explist] [‘;’]
		*/

		skipSpace(in);

		Block ret{};
		ret.start = in.getLoc();

		while (true)
		{
			const char ch = in.peek();

			if (ch == 'r')
			{
				if (checkReadTextToken(in, "return"))
				{
					ret.hadReturn = true;

					skipSpace(in);

					const char ch1 = in.peek();

					if (ch1 == ';')
						in.skip();//thats it
					else if (!isBasicBlockEnding(in, ch1))
					{
						ret.retExprs = readExpList(in);
						readOptToken(in, ";");
					}
					break;// no more loop
				}
			}
			else if (isBasicBlockEnding(in, ch))
				break;// no more loop

			// Not some end / return keyword, must be a statement

			ret.statList.push_back(readStatment(in));
		}
		ret.end = in.getLoc();
		return ret;
	}

	inline Function readFuncBody(AnyInput auto& in)
	{
		/*
			funcbody ::= ‘(’ [parlist] ‘)’ block end
			parlist ::= namelist [‘,’ ‘...’] | ‘...’
		*/
		Function ret{};
		ret.place = in.getLoc();

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
			ret.params.emplace_back(readName(in));

			while (checkReadToken(in, ","))
			{
				if (checkReadToken(in, "..."))
				{
					ret.hasVarArgParam = true;
					break;//cant have anything after the ... arg
				}
				ret.params.emplace_back(readName(in));
			}
		}

		requireToken(in, ")");
		ret.block = readBlock(in);
		requireToken(in, "end");
	}

	inline std::string readLabel(AnyInput auto& in)
	{
		//label ::= ‘::’ Name ‘::’

		requireToken(in, "::");

		const std::string res = readName(in);

		requireToken(in, "::");

		return res;
	}

	inline Block readDoEndBlock(AnyInput auto& in)
	{
		requireToken(in, "do");
		Block bl = readBlock(in);
		requireToken(in, "end");

		return bl;
	}

	inline Statement readStatment(AnyInput auto& in)
	{
		/*
		 stat ::=  ‘;’ |
		 varlist ‘=’ explist |
		 functioncall |
		*/

		skipSpace(in);

		Statement ret;
		ret.place = in.getLoc();

		switch (in.peek())
		{
		case ';':
			ret.data = StatementType::SEMICOLON();
			return ret;

		case ':'://must be label
			ret.data = StatementType::LABEL(readLabel(in));
			return ret;

		case 'f'://for?,func?
			if (checkReadTextToken(in, "for"))
			{
				/*
				 for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end |
				 for namelist in explist do block end |
				*/

				NameList names = readNameList(in);

				if (names.size() == 1 && checkReadToken(in, "="))//1 name, then MAYBE equal
				{
					StatementType::FOR_LOOP_NUMERIC res{};
					res.varName = names[0];

					// for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end | 
					res.start = readExpr(in);
					requireToken(in, ",");
					res.end = readExpr(in);

					if (checkReadToken(in, ","))
						res.step = readExpr(in);

					res.bl = readDoEndBlock(in);

					ret.data = res;
					return ret;
				}
				// Generic Loop
				// for namelist in explist do block end | 

				StatementType::FOR_LOOP_GENERIC res{};
				res.varNames = names;

				requireToken(in, "in");
				res.exprs = readExpList(in);
				res.bl = readDoEndBlock(in);

				ret.data = res;
				return ret;
			}
			if (checkReadTextToken(in, "function"))
			{ // function funcname funcbody
				StatementType::FUNCTION_DEF res{};

				res.name = readFuncName(in);
				res.func = readFuncBody(in);

				ret.data = res;
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
					StatementType::LOCAL_FUNCTION_DEF res;
					res.name = readName(in);
					res.func = readFuncBody(in);

					ret.data = res;
					return ret;
				}
				// Local Variable

				StatementType::LOCAL_ASSIGN res;
				res.name = readAttNameList(in);

				if (checkReadToken(in, "="))
				{// [‘=’ explist]
					res.exprs = readExpList(in);
				}
				ret.data = res;
				return ret;
			}
			break;
		case 'd'://do?
			if (checkReadTextToken(in, "do")) // do block end
			{
				Block bl = readBlock(in);
				requireToken(in, "end");
				ret.data = StatementType::DO_BLOCK(bl);
				return ret;
			}
			break;
		case 'b'://break?
			if (checkReadTextToken(in, "break"))
			{
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
				Expression expr = readExpr(in);
				Block bl = readDoEndBlock(in);
				ret.data = StatementType::WHILE_LOOP(expr, bl);
				return ret;
			}
			break;
		case 'r'://repeat?
			if (checkReadTextToken(in, "repeat"))
			{ // repeat block until exp
				Block bl = readBlock(in);
				requireToken(in, "until");
				Expression expr = readExpr(in);

				ret.data = StatementType::REPEAT_UNTIL(expr, bl);
				return ret;
			}
			break;
		case 'i'://if?
			if (checkReadTextToken(in, "if"))
			{ // if exp then block {elseif exp then block} [else block] end

				StatementType::IF_THEN_ELSE res{};

				res.cond = readExpr(in);

				requireToken(in, "then");

				res.bl = readBlock(in);

				while (checkReadTextToken(in, "elseif"))
				{
					Expression elExpr = readExpr(in);
					requireToken(in, "then");
					Block elBlock = readBlock(in);

					res.elseIfs.push_back({ elExpr,elBlock });
				}

				if (checkReadTextToken(in, "else"))
					res.elseBlock = readBlock(in);

				requireToken(in, "end");

				ret.data = res;
				return ret;
			}
			break;

		default://none of the above...
			break;
		}
		//try assign or func-call
	}

	struct ParsedFile
	{
		//TypeList types
		Block code;
	};

	inline ParsedFile parseFile(AnyInput auto& in)
	{
		return { readBlock(in) };
	}
}