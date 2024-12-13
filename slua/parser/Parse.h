/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>

#include "SkipSpace.h"
#include "RequireToken.h"
#include "ReadName.h"
#include "ReadOperators.h"

namespace sluaParse
{
	inline Expression readExpr(AnyInput auto& in)
	{
		/*
		nil | false | true | Numeral | LiteralString | ‘...’ | functiondef | 
		 prefixexp | tableconstructor | exp binop exp | unop exp 
		*/

		Expression res;

		res.unOp = readOptUnOp(in);

		skipSpace(in);

		switch (in.peek())
		{
		case 'n':
			if (checkReadTextToken(in, "nil"))
			{
				//the in args
				break;
			}
			break;
		case 'f':

			if (checkReadTextToken(in, "false")) { break; }
			if (checkReadTextToken(in, "function")) { break; }
			break;
		case 't':
			if (checkReadTextToken(in, "true")) { break; }
			break;
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
			//numeral

			break;
		case '"':
		case '\'':
		case '[':
			//literal?
			break;
		case '.':
			if (checkReadToken(in, "..."))
			{
				//the in args
				break;
			}
			break;
		case '(':
			//prefixexp
			break;
		case '{':
			//tableconstructor
			break;
		}

		//check bin op


		const BinOpType binOp = readOptBinOp(in);

		if (binOp == BinOpType::NONE)
			return res;
	}

	inline std::string readLabel(AnyInput auto& in)
	{
		//label ::= ‘::’ Name ‘::’

		requireToken(in, "::");

		const std::string res = readName(in);

		requireToken(in, "::");

		return res;
	}

	inline Statement readStatment(AnyInput auto& in)
	{
		/*
		 stat ::=  ‘;’ |
		 varlist ‘=’ explist |
		 functioncall |
		 label |
		 break |
		 goto Name |
		 do block end |
		 while exp do block end |
		 repeat block until exp |
		 if exp then block {elseif exp then block} [else block] end |
		 for Name ‘=’ exp ‘,’ exp [‘,’ exp] do block end |
		 for namelist in explist do block end |
		 function funcname funcbody |
		 local function Name funcbody |
		 local attnamelist [‘=’ explist]
		*/

		skipSpace(in);

		switch (in.peek())
		{
		case ';':
			return { StatementData::with<StatementType::SEMICOLON>(),in.getLoc() };

		case ':'://must be label
			return { StatementData::with<StatementType::LABEL>(readLabel(in)),in.getLoc() };

		case 'f'://for?,func?
			if (checkReadTextToken(in, "for"))
			{
				requireToken(in, "end");
				break;//TODO: replace with return
			}
			if (checkReadTextToken(in, "function"))
			{
				break;//TODO: replace with return
			}

			break;
		case 'l'://local?
			if (checkReadTextToken(in, "local"))
			{//func, or var
				if (checkReadTextToken(in, "function"))
				{
					break;//TODO: replace with return
				}
				//var

				break;//TODO: replace with return
			}
			break;
		case 'd'://do?
			if (checkReadTextToken(in, "do")) // do block end
			{
				//TODO

				requireToken(in, "end");
			}
			break;
		case 'b'://break?
			if (checkReadTextToken(in, "break"))
				return { StatementData::with<StatementType::BREAK>(),in.getLoc() };
			break;
		case 'g'://goto?
			if (checkReadTextToken(in, "goto"))//goto Name
				return { StatementData::with<StatementType::GOTO>(readName(in)),in.getLoc() };
			break;
		case 'w'://while?
			if (checkReadTextToken(in, "while"))
			{
				Expression expr = readExpr(in);

				requireToken(in, "end");
				break;//TODO: replace with return
			}
			break;
		case 'r'://repeat?
			if (checkReadTextToken(in, "repeat"))
			{
				break;//TODO: replace with return
			}
			break;
		case 'i'://if?
			if (checkReadTextToken(in, "if"))
			{
				requireToken(in, "end");
				break;//TODO: replace with return
			}
			break;

		default://none of the above...
			break;
		}
		//try assign or func-call
	}
	inline Block readBlock(AnyInput auto& in)
	{
		//block ::= {stat} [retstat]

		//0+ stat
		//0/1 return
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