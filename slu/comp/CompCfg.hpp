/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>

#include <slu/lang/BasicState.hpp>

namespace slu::comp
{
	//
	// All views / spans are only required to live as long as the call.
	// The compiler wont emit `../`, so there is no need to sanitize them for that (except by you in rootPaths)
	// Your paths must not contain `\`
	// If one of your callbacks is thread-unsafe, add some mutex, because all of them will be called by many threads.
	//



	//TODO: implement compiler plugins, maybe redesign the api too.

	//Used to implement external namespaces (lua::ext::*)
	struct CompPlugin
	{
		// If it would need to call to some dll, or external code, it is likely not safe
		bool comptimeSafe : 1 = false;

		lang::ModPathView root;

		//TODO: choose a type for the list (string?) (mod path?)
		using GetItemListFn = std::vector<int>(*)(const lang::ModPathView& path);
		GetItemListFn getItemListPtr;

		using ItemBoolFn = bool(*)(const lang::ModPathView& path);
		ItemBoolFn itemExistsPtr;
	};
	struct TmpFile
	{
		std::string realPath;
		using DestroyFn = void (*)(TmpFile& thiz);
		DestroyFn destroy = [](TmpFile& thiz) {};

		constexpr TmpFile(std::string&& path, DestroyFn destroy)
			: realPath(path), destroy(destroy) {}
		constexpr TmpFile(DestroyFn destroy)
			: destroy(destroy) {};
		constexpr TmpFile() = default;

		~TmpFile() { destroy(*this); };
		//Move only:
		void release() {
			realPath.clear();
		}
		TmpFile(TmpFile&& o) noexcept {
			*this = std::move(o);
		}
		TmpFile& operator=(TmpFile&& o) noexcept {
			*this = *&o;
			o.release();
			return *this;
		}
	private:
		TmpFile(const TmpFile&) = default;
		TmpFile& operator=(const TmpFile&) = default;
	};
	struct CompCfg
	{
		//you need to add a newline yourself, if you need to (use println or something)
		//Note: msg may contain nulls
		using LogFn = void(*)(const std::string_view msg);
		LogFn logPtr;//Warnings, info, dbg.
		LogFn errPtr;

		//If empty, then dont create it yet.
		using MkTmpFileFn = TmpFile(*)(std::optional<std::span<const uint8_t>> contents);
		MkTmpFileFn mkTmpFilePtr;

		using GetFileContentsFn = std::optional<std::vector<uint8_t>>(*)(const std::string_view path);
		GetFileContentsFn getFileContentsPtr;

		using FileBoolFn = bool(*)(const std::string_view path);
		FileBoolFn fileExistsPtr;
		FileBoolFn isFolderPtr;
		FileBoolFn isSymLinkPtr;

		//file paths inside a path, return empty list if path is invalid or doesnt exist
		using GetFileListFn = std::vector<std::string>(*)(const std::string_view path);
		GetFileListFn getFileListPtr;
		GetFileListFn getFileListRecPtr;//recursive version

		std::span<const CompPlugin> plugins;

		std::span<const std::string> rootPaths;//All of them must have some crate.

		//will use exactly that many threads, not any less, not any more
		// (other than the main thread which is not counted here)
		size_t extraThreadCount = 1;
		/*(size_t)std::max(//max, so no negative thread counts
			int64_t(std::thread::hardware_concurrency())
			- 3,//2 sys threads, 1 for main thread
			0LL
		);*/
	};
}