

local fieldTest = {
	apple,
	syn = 4,
	syn --[[]]= 3,
	syn = --[[]]2,
	syn --[[]]= --[[]]1,

	synW--[[]]= 3,
	synW=--[[]]2,
	synW--[[]]=--[[]]1,

	
	sgnl --[[
	]]= 3,
	sgnl = --[[
	]]2,
	sgnl --[[
	]]= --[[
	]]1,

	sgnlW--[[
	]]= 3,
	sgnlW=--[[
	]]2,
	sgnlW--[[
	]]=--[[
	]]1,
	
	
	sgn --comment
	= 3,
	sgn = --comment
	2,
	sgn --comment
	= --comment
	1,

	sgnW--comment
	= 3,
	sgnW=--comment
	2,
	sgnW--comment
	=--comment
	1,

	"",

}