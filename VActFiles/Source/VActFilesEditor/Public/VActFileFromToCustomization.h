#pragma once

#include "CoreMinimal.h"
#include "VActFileTypes.h"

#include "IDetailCustomization.h"
#include "Input/Reply.h"

#include "DetailCategoryBuilder.h"
#include "DetailLayoutBuilder.h"
#include "DetailWidgetRow.h"

#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SFilePathPicker.h"
#include "Widgets/Input/SDirectoryPicker.h"
#include "Widgets/Input/SComboBox.h"

#include "VActFileUtils.h"
#include "VActFiles.h"

#define LOCTEXT_NAMESPACE "VActFileFromToCustomization"

template<typename T, typename TypeDetailCustomization>
class FVActFileFromToCustomization : public IDetailCustomization
{
public:
	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override
	{
		// Get selected object
		TArray<TWeakObjectPtr<UObject>> Objects;
		DetailBuilder.GetObjectsBeingCustomized(Objects);
		if (Objects.Num() == 1)
		{
			Context = Cast<T>(Objects[0].Get());
			if (Context.IsValid())
			{
				if (FVActFileUtils::FormatOptions.IsValidIndex((int32)Context->IOInfo.FromFormat)) { FromFormat = Context->IOInfo.FromFormat; }
				if (FVActFileUtils::FormatOptions.IsValidIndex((int32)Context->IOInfo.ToFormat)) { ToFormat = Context->IOInfo.ToFormat; }
				FromFilePath = Context->IOInfo.FromFilePath;
				ToFilePath = Context->IOInfo.ToFilePath;
				bStrict = Context->IOInfo.bStrict;
			}
		}

		auto _FormatGenerator = [](TSharedPtr<FString> InItem) { return SNew(STextBlock).Text(FText::FromString(*InItem)); };

		IDetailCategoryBuilder& Category = DetailBuilder.EditCategory(TypeDetailCustomization::EditCategory);

		const FText LoadButton = LOCTEXT("LoadButtonRow", "Load From File");
		const FText CheckBox = LOCTEXT("LoadPropRow", "Strict Read/Write");
		Category.AddCustomRow(LoadButton)
		.NameContent()
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth()
			[
				SNew(SButton).VAlign(VAlign_Center)
				.OnClicked(this, &FVActFileFromToCustomization::OnClickLoad)
				.Content()[SNew(STextBlock).Text(LoadButton)].HAlign(HAlign_Center)
			]
			+ SHorizontalBox::Slot().AutoWidth()
			[
				SNew(SComboBox<TSharedPtr<FString>>)
				.OptionsSource(&FVActFileUtils::FormatOptions)
				.OnGenerateWidget_Lambda(_FormatGenerator)
				.OnSelectionChanged(this, &FVActFileFromToCustomization::OnFromFormatSelectionChanged)
				.InitiallySelectedItem(FVActFileUtils::FormatOptions[(int32)FromFormat])
				.Content()
				[
					SNew(STextBlock)
					.Text(this, &FVActFileFromToCustomization::GetFromFormat)
				]
			]
			+ SHorizontalBox::Slot().AutoWidth()
			[
				SNew(SCheckBox)
				.IsChecked((ECheckBoxState)bStrict)
				.OnCheckStateChanged(this, &FVActFileFromToCustomization::OnCheckStateChangedStrict)
				.Content()[SNew(STextBlock).Text(CheckBox)]
			]
		].ValueContent()
		.VAlign(VAlign_Center)
		[
			SNew(SFilePathPicker)
			.BrowseButtonImage(FAppStyle::GetBrush("PropertyWindow.Button_Ellipsis"))
			.BrowseButtonStyle(FAppStyle::Get(), "HoverHintOnly")
			.BrowseButtonToolTip(FText::FromString("Choose a file"))
			.BrowseDirectory(FPaths::ProjectContentDir())
			.FilePath(this, &FVActFileFromToCustomization::GetFromFilePath)
			.FileTypeFilter(TypeDetailCustomization::TypeFilter)
			.OnPathPicked(this, &FVActFileFromToCustomization::OnFromFilePathPicked)
		];

		const FText SaveButton = LOCTEXT("SaveButtonRow", "Save To File");
		Category.AddCustomRow(SaveButton)
		.NameContent()
		[
		SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth()
		[
			SNew(SButton).VAlign(VAlign_Center)
			.OnClicked(this, &FVActFileFromToCustomization::OnClickSave)
			.Content()[SNew(STextBlock).Text(SaveButton)].HAlign(HAlign_Center)
		]
		+ SHorizontalBox::Slot().AutoWidth()
		[
			SNew(SComboBox<TSharedPtr<FString>>)
			.OptionsSource(&FVActFileUtils::FormatOptions)
			.OnGenerateWidget_Lambda(_FormatGenerator)
			.OnSelectionChanged(this, &FVActFileFromToCustomization::OnToFormatSelectionChanged)
			.InitiallySelectedItem(FVActFileUtils::FormatOptions[(int32)ToFormat])
			.Content()
			[
				SNew(STextBlock)
					.Text(this, &FVActFileFromToCustomization::GetToFormat)
			]
		]
		].ValueContent()
		.VAlign(VAlign_Center)
		[
			SNew(SFilePathPicker)
			.BrowseButtonImage(FAppStyle::GetBrush("PropertyWindow.Button_Ellipsis"))
			.BrowseButtonStyle(FAppStyle::Get(), "HoverHintOnly")
			.BrowseButtonToolTip(FText::FromString("Choose a file"))
			.BrowseDirectory(FPaths::ProjectContentDir())
			.FilePath(this, &FVActFileFromToCustomization::GetToFilePath)
			.FileTypeFilter(TypeDetailCustomization::TypeFilter)
			.OnPathPicked(this, &FVActFileFromToCustomization::OnToFilePathPicked)
		];
	}

	static void LoadFromFile(T* InContext, const TCHAR* FilePath, EVActFileFormat Format, bool bStrict = true)
	{
		bool bSuccess = false;
		FVActParseRoot Root(Format);

		switch (Format)
		{
		case EVActFileFormat::Json: bSuccess = FVActFileJson::Load(Root, FilePath)
			&& (bStrict 
				? TypeDetailCustomization::ParseFromCursorsJsonStrict(InContext, Root) 
				: TypeDetailCustomization::ParseFromCursorsJson(InContext, Root)); break;
		case EVActFileFormat::Compact: bSuccess = FVActFileCompact::Load(Root, FilePath)
			&& TypeDetailCustomization::ParseFromCursorsCompact(InContext, Root); break;
		case EVActFileFormat::Binary: bSuccess = FVActFileBinary::Load(Root, FilePath)
			&& TypeDetailCustomization::ParseFromCursorsBinary(InContext, Root); break;
		}
	}

	static void SaveToFile(T* InContext, const TCHAR* FilePath, EVActFileFormat Format, bool bStrict = true)
	{
		bool bSuccess = false;
		FVActEmitRoot Root(Format);

		switch (Format)
		{
		case EVActFileFormat::Json: bSuccess = true
			&& (bStrict
				? TypeDetailCustomization::EmitToCursorsJsonStrict(InContext, Root)
				: TypeDetailCustomization::EmitToCursorsJson(InContext, Root))
			&& FVActFileJson::Save(Root, FilePath); break;
		case EVActFileFormat::Compact: bSuccess = TypeDetailCustomization::EmitToCursorsCompact(InContext, Root)
			&& FVActFileCompact::Save(Root, FilePath); break;
		case EVActFileFormat::Binary: bSuccess = TypeDetailCustomization::EmitToCursorsBinary(InContext, Root)
			&& FVActFileBinary::Save(Root, FilePath); break;
		}
	}

	FReply OnClickLoad()
	{
		const bool bValid = Context.IsValid() && !FromFilePath.IsEmpty();
		if (bValid)
		{
			LoadFromFile(Context.Get(), *FromFilePath, FromFormat, bStrict);
		}
		return FReply::Handled();
	}

	void OnFromFilePathPicked(const FString& InPath)
	{
		FromFilePath = InPath;
		if (Context.IsValid())
		{
			Context->Modify();
			Context->IOInfo.FromFilePath = InPath;
		}
	}

	FString GetFromFilePath() const
	{
		return Context.IsValid() ? Context->IOInfo.FromFilePath : FromFilePath;
	}

	FReply OnClickSave()
	{
		const bool bValid = Context.IsValid() && !ToFilePath.IsEmpty();
		if (bValid)
		{
			SaveToFile(Context.Get(), *ToFilePath, ToFormat, bStrict);
		}
		return FReply::Handled();
	}


	void OnToFilePathPicked(const FString& InPath)
	{
		ToFilePath = InPath;
		if (Context.IsValid())
		{
			Context->Modify();
			Context->IOInfo.ToFilePath = InPath;
		}
	}

	FString GetToFilePath() const
	{
		return Context.IsValid() ? Context->IOInfo.ToFilePath : ToFilePath;
	}

	void OnFromFormatSelectionChanged(TSharedPtr<FString> NewSelection, ESelectInfo::Type)
	{
		FromFormat = FVActFileUtils::MapFormat[NewSelection];
		if (Context.IsValid())
		{
			Context->Modify();
			Context->IOInfo.FromFormat = FromFormat;
		}
	}

	FText GetFromFormat() const
	{
		const int32 _Format = (int32)(Context.IsValid() ? Context->IOInfo.FromFormat : FromFormat);
		const bool bValid = FVActFileUtils::FormatOptions.IsValidIndex(_Format);
		return bValid ? FText::FromString(*FVActFileUtils::FormatOptions[_Format].Get()) : FText::GetEmpty();
	}

	void OnToFormatSelectionChanged(TSharedPtr<FString> NewSelection, ESelectInfo::Type)
	{
		ToFormat = FVActFileUtils::MapFormat[NewSelection];
		if (Context.IsValid())
		{
			Context->Modify();
			Context->IOInfo.ToFormat = ToFormat;
		}
	}

	FText GetToFormat() const
	{
		const int32 _Format = (int32)(Context.IsValid() ? Context->IOInfo.ToFormat : ToFormat);
		const bool bValid = FVActFileUtils::FormatOptions.IsValidIndex(_Format);
		return bValid ? FText::FromString(*FVActFileUtils::FormatOptions[_Format].Get()) : FText::GetEmpty();
	}

	FVActFileFromToCustomization()
	{
		ToFormat = FromFormat = EVActFileFormat::Json;
	}

	void OnCheckStateChangedStrict(ECheckBoxState NewState)
	{
		bStrict = (bool)NewState;
		if (Context.IsValid())
		{
			Context->Modify();
			Context->IOInfo.bStrict = bStrict;
		}
	}

protected:
	TWeakObjectPtr<T> Context;

	FString FromFilePath;

	FString ToFilePath;

	EVActFileFormat FromFormat;

	EVActFileFormat ToFormat;

	bool bStrict;

};

#undef LOCTEXT_NAMESPACE