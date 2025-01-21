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

#include "ReadName.hpp"
#include "RequireToken.hpp"
#include "SkipSpace.hpp"
#include "ReadExpr.hpp"



/*

	TODO:

	[X] chunk ::= block

	[~] block ::= {stat} [retstat]

	[~] stat ::= [X] ‘;’ |
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
		[_] function funcname funcbody |
		[_] local function Name funcbody |
		[~] local attnamelist [‘=’ explist]

	[_] attnamelist ::=  Name attrib {‘,’ Name attrib}

	[_] attrib ::= [‘<’ Name ‘>’]

	[~] retstat ::= return [explist] [‘;’]

	[X] label ::= ‘::’ Name ‘::’

	[_] funcname ::= Name {‘.’ Name} [‘:’ Name]

	[_] varlist ::= var {‘,’ var}

	[_] var ::=  Name | prefixexp ‘[’ exp ‘]’ | prefixexp ‘.’ Name

	[X] namelist ::= Name {‘,’ Name}

	[X] explist ::= exp {‘,’ exp}

	[_] exp ::=  (~)nil | (~)false | (~)true | Numeral | LiteralString | (~)‘...’ | functiondef |
		 prefixexp | tableconstructor | [X] exp binop exp | [X] unop exp

	[_] prefixexp ::= var | functioncall | ‘(’ exp ‘)’

	[_] functioncall ::=  prefixexp args | prefixexp ‘:’ Name args

	[_] args ::=  ‘(’ [explist] ‘)’ | tableconstructor | LiteralString

	[_] functiondef ::= function funcbody

	[_] funcbody ::= ‘(’ [parlist] ‘)’ block end

	[_] parlist ::= namelist [‘,’ ‘...’] | ‘...’

	[_] tableconstructor ::= ‘{’ [fieldlist] ‘}’

	[_] fieldlist ::= field {fieldsep field} [fieldsep]

	[_] field ::= ‘[’ exp ‘]’ ‘=’ exp | Name ‘=’ exp | exp

	[X] fieldsep ::= ‘,’ | ‘;’

	[X] binop ::=  ‘+’ | ‘-’ | ‘*’ | ‘/’ | ‘//’ | ‘^’ | ‘%’ |
		 ‘&’ | ‘~’ | ‘|’ | ‘>>’ | ‘<<’ | ‘..’ |
		 ‘<’ | ‘<=’ | ‘>’ | ‘>=’ | ‘==’ | ‘~=’ |
		 and | or

	[X] unop ::= ‘-’ | not | ‘#’ | ‘~’

*/


namespace sluaParse
{
	inline Block readBlock(AnyInput auto& in)
	{
		/*
			block ::= {stat} [retstat]
			retstat ::= return [explist] [‘;’]
		*/

		skipSpace(in);

		Block ret{};
		ret.start = in.getLoc();

		//TODO: implement
		//0+ stat
		/*
		TODO: check for the following, to allow for 0 statements
			end
			until
			elseif
			else
			return
		*/

		if (checkReadTextToken(in, "return"))
		{
			ret.hadReturn = true;
			/*
			TODO: check for the following, to allow for empty returns
				end
				until
				elseif
				else
				';'
			*/
			ret.retExprs = readExpList(in);
			readOptToken(in, ";");
		}

		ret.end = in.getLoc();
		return ret;
	}

	inline Function readFuncBody(AnyInput auto& in)
	{
		/*
			block ::= {stat} [retstat]
			retstat ::= return [explist] [‘;’]
		*/
		Function ret{};
		ret.start = in.getLoc();

		return ret;
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
				break;//TODO: replace with return
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
					break;//TODO: replace with return
				}
				// Local Variable

				StatementType::LOCAL_ASSIGN res;

				//TODO: attnamelist

				if (checkReadToken(in, "="))
				{ // [‘=’ explist]
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